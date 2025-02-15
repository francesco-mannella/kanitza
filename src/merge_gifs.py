from PIL import Image


def merge_gifs(first_frames, second_frames, merged_gif, frame_duration=2000):

    # Prepare a list to store the merged frames
    frames = []

    # Iterate over the frames and merge them
    for frame1, frame2 in zip(first_frames, second_frames):
        # Ensure both frames have the same size
        width, height = frame1.size

        # Create a new image for this frame
        merged_frame = Image.new('RGB', (width, height * 2))

        # Paste the frames from the first and second GIF
        merged_frame.paste(frame1, (0, 0))
        merged_frame.paste(frame2, (0, height))

        # Append the merged frame to the list of frames
        frames.append(merged_frame)

    gif_file = f'{merged_gif}.gif'    
    # Save the frames as a new GIF
    frames[0].save(
        gif_file,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=frame_duration
    )

    print(f'save {gif_file}')
