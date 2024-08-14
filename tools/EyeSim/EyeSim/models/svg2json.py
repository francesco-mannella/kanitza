#!/usr/bin/python3
import matplotlib.pyplot as plt
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
import json
import numpy as np


class svn2jsonConverter:
    def __init__(
        self,
        fileroot,
        env_scale=1,
        object_colors=None,
        object_types=None,
        object_masses=None,
        joint_torques=None,
    ):

        self.fileroot = fileroot
        self.env_scale = env_scale
        self.object_colors = object_colors
        self.object_types = object_types
        self.object_masses = object_masses
        self.joint_torques = joint_torques

        self.getDoc()

    def getDoc(self):

        # read the SVG file
        self.doc = minidom.parse(self.fileroot + ".svg")

        paths = {}
        joints = {}
        allpoints = []

        for path in self.doc.getElementsByTagName("path"):
            idf = path.getAttribute("id")
            d = path.getAttribute("d")
            points = []
            for e in parse_path(d):
                if isinstance(e, Line):
                    x0 = e.start.real
                    y0 = e.start.imag
                    x1 = e.end.real
                    y1 = e.end.imag
                    points.append([x0, y0])
            #
            points.append([x1, y1])
            points.append(points[0])
            paths[idf] = np.array(points)
            allpoints.append(paths[idf])

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
            x, y, vx, vy = self.rescale(name)

            obj_dict = {
                "angle": 0.0,
                "name": obj,
                "color": self.object_colors[obj],
                "position": {"x": x, "y": y},
                "type": self.object_types[obj],
                "fixture": [
                    {
                        "density": self.object_masses[obj],
                        "group_index": 0,
                        "polygon": {"vertices": {"x": vx.tolist(), "y": vy.tolist()}},
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

                # point c wrt A
                *_, cAx, cAy = self.rescale(nameA, np.array([[cx, cy]]))
                # point c wrt B
                *_, cBx, cBy = self.rescale(nameB, np.array([[cx, cy]]))

                torque = self.joint_torques[nameA + "_to_" + nameB]

                ulimit = 0
                llimit = 0

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
                    "maxMotorTorque": self.joint_torques[nameA + "_to_" + nameB],
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

        self.jsn = json.dumps(self.world, indent=4)
        if root is None:
            root = self.fileroot
        with open(root + ".json", "w") as json_file:
            json_file.write(self.jsn)

    def rescale(self, name, other_points=None):
        points = self.paths[name] - self.min
        points[:, 1] = self.range[1] - points[:, 1]
        x, y = points.min(0)
        vx, vy = (points - [[x, y]]).T
        x, y = points.min(0)
        if other_points is not None:
            op = other_points - self.min
            op[:, 1] = self.range[1] - op[:, 1]
            vox, voy = (op - [[x, y]]).T
            return x, y, vx, vy, vox[0], voy[0]
        else:
            return x, y, vx, vy


if __name__ == "__main__":

    n_sensors = 40

    object_colors = {
        "triangle": [0.7, 0.7, 0.7],
    }


    object_masses = {
        "triangle": 0.0,
    }

    object_types = {
        "triangle": 0,
    }

    joint_torques = {
    }

    conv = svn2jsonConverter(
        "eyesim",
        env_scale=1,
        object_colors=object_colors,
        object_types=object_types,
        object_masses=object_masses,
        joint_torques=joint_torques,
    )

    objs = [
        "eyesim",
    ]

    for obj in objs:
        print("\n\n\n\n\n")
        conv.buildWorld(
            exclude_objs=[x for x in objs if x != obj],
        )
        conv.saveWorld(root=obj)
