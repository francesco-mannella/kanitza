from PIL import Image, ImageSequence

def merge_gifs(first_gif, second_gif, merged_gif, frame_duration=300):
    # Open the GIFs
    first = Image.open(first_gif)
    second = Image.open(second_gif)

    # Prepare a list to store the merged frames
    frames = []

    # Iterate over the frames and merge them
    for frame1, frame2 in zip(ImageSequence.Iterator(first), ImageSequence.Iterator(second)):
        # Ensure both frames have the same size
        width, height = frame1.size
        
        # Create a new image for this frame
        merged_frame = Image.new('RGB', (width, height * 2))

        # Paste the frames from the first and second GIF
        merged_frame.paste(frame1, (0, 0))
        merged_frame.paste(frame2, (0, height))

        # Append the merged frame to the list of frames
        frames.append(merged_frame)

    # Save the frames as a new GIF
    frames[0].save(
        merged_gif,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=frame_duration
    )
    print(merged_gif)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("first_gif", help="path to first gif file")
parser.add_argument("second_gif", help="path to second gif file")
args = parser.parse_args()

first_gif = args.first_gif
second_gif = args.second_gif
merged_gif = first_gif.replace('maps', 'merged').replace('sim', 'merged')

merge_gifs(first_gif, second_gif, merged_gif)

