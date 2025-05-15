#!/usr/bin/python3
from svg.path import parse_path
from xml.dom import minidom
import json
import numpy as np
import argparse


class svn2jsonConverter:
    """
    Converts SVG files to JSON format for physics simulations.

    This class parses an SVG file, extracts path and joint information,
    and converts it into a JSON format that can be used to define a
    world in a physics engine.
    """

    def __init__(
        self,
        fileroot,
        env_scale=1,
        object_colors=None,
        object_types=None,
        object_masses=None,
        joint_torques=None,
        object_zorders=None,
    ):
        """
        Initializes the svn2jsonConverter.

        Args:
            fileroot (str): Base name of the SVG file (without extension).
            env_scale (float, optional): Scaling factor. Defaults to 1.
            object_colors (dict, optional): Object colors. Defaults to None.
            object_types (dict, optional): Object types. Defaults to None.
            object_masses (dict, optional): Object masses. Defaults to None.
            joint_torques (dict, optional): Joint torques. Defaults to None.
            object_zorders (dict, optional): Object z-orders. Defaults to None.
        """
        self.fileroot = fileroot
        self.env_scale = env_scale
        self.object_colors = object_colors
        self.object_types = object_types
        self.object_masses = object_masses
        self.joint_torques = joint_torques
        self.object_zorders = object_zorders

        self.getDoc()

    def getDoc(self):
        """
        Parses the SVG file to extract path and joint information.

        Reads the SVG file, extracts path data for each object, and
        identifies joint connections between objects. Stores the
        extracted information in the class's attributes.
        """
        # Read the SVG file
        self.doc = minidom.parse(self.fileroot)

        paths = {}
        joints = {}
        allpoints = []

        # Iterate over each path element in the SVG
        for path in self.doc.getElementsByTagName("path"):
            idf = path.getAttribute("id")
            if idf not in self.object_colors:
                continue
            d = path.getAttribute("d")
            points = []
            # Parse the path data
            for e in parse_path(d):
                x0 = e.start.real
                y0 = e.start.imag
                x1 = e.end.real
                y1 = e.end.imag
                points.append([x0, y0])
            # Append the last point and the first point to close the loop
            points.append([x1, y1])
            points.append(points[0])
            paths[idf] = np.array(points)
            allpoints.append(paths[idf])

        # Iterate over each circle/ellipse element (representing joints)
        for joint in self.doc.getElementsByTagName(
            "circle"
        ) + self.doc.getElementsByTagName("ellipse"):
            idf = joint.getAttribute("id")

            objA, objB = idf.split("_to_")
            cx = abs(float(joint.getAttribute("cx")))
            cy = abs(float(joint.getAttribute("cy")))

            jInfo = (objA, objB, cx, cy)

            if objA in joints.keys():
                joints[objA].append(jInfo)
            else:
                joints[objA] = [jInfo]

        self.doc.unlink()

        self.paths = paths
        self.joints = joints
        self.allpoints = np.vstack(allpoints)
        self.min = self.allpoints.min(0)
        self.max = self.allpoints.max(0)
        self.mean = self.allpoints.mean(0)
        self.range = self.max - self.min

    def buildWorld(self, exclude_objs=None):
        """
        Constructs the world dictionary for the physics engine.

        Creates a dictionary representing the world, including gravity,
        object properties (position, shape, color, mass), and joint
        constraints. Uses data from the SVG file and object properties.

        Args:
            exclude_objs (list, optional): Objects to exclude.
            Defaults to None.
        """
        self.world = {
            "gravity": {"y": 0, "x": 0},
            "autoClearForces": True,
            "continuousPhysics": True,
            "subStepping": False,
            "warmStarting": True,
        }

        self.allpoints = np.vstack(self.allpoints)

        self.world["body"] = []
        for obj in self.paths.keys():
            name = obj
            x, y, vx, vy = self.get_position_and_vertices(name)

            obj_dict = {
                "angle": 0.0,
                "name": obj,
                "color": self.object_colors[obj],
                "position": {"x": x, "y": y},
                "type": self.object_types[obj],
                "zorder": self.object_zorders[obj],
                "fixture": [
                    {
                        "density": self.object_masses[obj],
                        "group_index": 0,
                        "polygon": {
                            "vertices": {"x": vx.tolist(), "y": vy.tolist()}
                        },
                    }
                ],
            }
            self.world["body"].append(obj_dict)
        self.world["body"] = [
            x for x in self.world["body"] if not x["name"] in exclude_objs
        ]

        body_idcs = {}
        print("--bodies--")
        for i, obj in enumerate(self.world["body"]):
            print(self.world["body"][i]["name"])
            body_idcs[self.world["body"][i]["name"]] = i
        print("----")

        self.world["joint"] = []
        for joint_set in self.joints.values():
            for joint in joint_set:
                nameA, nameB, cx, cy = joint

                # get_position_and_vertices joint coordinates relative to
                # bodies A and B
                *_, cAx, cAy = self.get_position_and_vertices(
                    nameA, np.array([[cx, cy]])
                )
                *_, cBx, cBy = self.get_position_and_vertices(
                    nameB, np.array([[cx, cy]])
                )

                torque = self.joint_torques[nameA + "_to_" + nameB]

                ulimit = 0
                llimit = 0

                # Set joint limits based on torque direction
                if torque > 0:
                    ulimit = 3.14
                    llimit = -3.14

                jont_dict = {
                    "name": nameA + "_to_" + nameB,
                    "type": "revolute",
                    "bodyA": body_idcs[nameA],
                    "bodyB": body_idcs[nameB],
                    "jointSpeed": 0,
                    "refAngle": 0,
                    "collideConnected": False,
                    "maxMotorTorque": self.joint_torques[
                        nameA + "_to_" + nameB
                    ],
                    "enableLimit": True,
                    "motorSpeed": 0,
                    "anchorA": {"x": cAx, "y": cAy},
                    "anchorB": {"x": cBx, "y": cBy},
                    "upperLimit": ulimit,
                    "lowerLimit": llimit,
                    "enableMotor": True,
                }

                self.world["joint"].append(jont_dict)

        for i, joint in enumerate(self.world["joint"]):
            print(self.world["joint"][i]["name"])

    def saveWorld(self, root=None):
        """
        Saves the constructed world dictionary to a JSON file.

        Args:
            root (str, optional): Root name for the output JSON file.
            If None, the fileroot attribute is used. Defaults to None.
        """
        self.jsn = json.dumps(self.world, indent=4)
        if root is None:
            root = self.fileroot
        with open(root + ".json", "w") as json_file:
            json_file.write(self.jsn)

    def get_position_and_vertices(self, name):
        """Returns the origin position (0,0) and vertices of a path.

        Args:
            name (str): The name of the path.

        Returns:
            tuple: A tuple containing the center position,
                   and the x and y coordinates of the vertices.
        """
        points = self.paths[name].T
        vx, vy = points
        x, y = 0, 0

        return x, y, vx, vy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SVG to JSON with object features."
        "Example usage: python script.py out.json obj1:1,0,0:1:10"
        " obj2:0,1,0:2:5",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "source_svg", type=str, help="Path to the source SVG file."
    )
    parser.add_argument(
        "destination_file", type=str, help="Path to the destination JSON file."
    )
    parser.add_argument(
        "objects",
        nargs="+",
        help="List of objects with features. " "Format: name:color:type:mass:zorder",
    )
    args = parser.parse_args()

    object_colors = {}
    object_masses = {}
    object_types = {}
    object_zorders = {}

    for obj_data in args.objects:
        name, color, obj_type, mass, zorder = obj_data.split(":")
        object_colors[name] = [float(c) for c in color.split(",")]
        object_types[name] = int(obj_type)
        object_masses[name] = float(mass)
        object_zorders[name] = int(zorder)

    conv = svn2jsonConverter(
        args.source_svg,
        env_scale=1,
        object_colors=object_colors,
        object_types=object_types,
        object_masses=object_masses,
        object_zorders=object_zorders,
    )

    objs = [obj_data.split(":")[0] for obj_data in args.objects]

    for obj in objs:
        print("\n\n\n\n\n")
        conv.buildWorld(
            exclude_objs=[x for x in objs if x not in conv.paths.keys()],
        )
    conv.saveWorld(root=args.destination_file.replace(".json", ""))
