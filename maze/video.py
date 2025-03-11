import cv2
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from environment import Position, Action
from q_table import QTable


class VideoRecorder:
    def __init__(self):
        self.grid_image = Image.open("images/grid.png")
        self.person_image = Image.open("images/person.png")

        self.grid_width, self.grid_height = self.grid_image.size

        self.cell_width = self.grid_width // 5
        self.cell_height = self.grid_height // 4

        # Resize the person image to fit within a grid cell
        self.person_size_ration = self.person_image.size[0] / self.person_image.size[1]
        self.person_height = 170
        self.person_image = self.person_image.resize((int(self.person_height * self.person_size_ration), self.person_height),
                                           Image.Resampling.LANCZOS)

        # Font settings for the additional text (ensure the font file exists or use a default)
        self.font = ImageFont.load_default(20)

        # Initialize video writer
        self.frame_rate = 1  # Frames per second
        self.output_video_path = "video.mp4"

        # Create video dimensions (grid + matrix side-by-side, with space for text below)
        self.info_space_height = 100  # Space below for additional information
        self.frame_width = self.grid_width * 2  # Grid width + matrix width
        self.frame_height = self.grid_height + self.info_space_height

        self.person_x_offsets = [5, -5, -20, -25, -30]
        self.person_y_offsets = [0, -28, -23, -15]

        # Initialize OpenCV video writer
        self.video = cv2.VideoWriter(
            self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.frame_rate, (self.frame_width, self.frame_height)
        )

    def add_frame(self, position: Position, q_table: QTable,
                  episode: int,
                  step: int,
                  epsilon: float,
                  prev_position: Optional[Position],
                  action: Optional[Action],
                  is_action_random: Optional[bool],
                  reward: Optional[float],
                  learning_rate: float,
                  discount_rate: float,
                  orig_current_action_value: Optional[float],
                  new_best_action_value: Optional[float],
                  new_current_action_value: Optional[float]):
        # Create a new frame with space for grid, matrix, and additional information
        frame_image = Image.new("RGB", (self.frame_width, self.frame_height), "white")
        frame_draw = ImageDraw.Draw(frame_image)

        # Paste the grid onto the frame
        frame_image.paste(self.grid_image, (0, 0))

        # Calculate the top-left corner of the cell for the person
        x_offset = position.col * self.cell_width + self.person_x_offsets[position.col]
        y_offset = position.row * self.cell_height + self.person_y_offsets[position.row]

        # Paste the person onto the grid
        frame_image.paste(self.person_image, (x_offset, y_offset), self.person_image)

        # Draw the 4x5 matrix to the right of the grid
        for r in range(4):
            for c in range(5):
                # Determine the value (1 for the current cell, 0 otherwise)
                value = 1 if (r == position.row and c == position.col) else 0

                # Calculate coordinates for the matrix cell
                top_left_x = self.grid_width + c * self.cell_width
                top_left_y = r *self. cell_height
                bottom_right_x = top_left_x + self.cell_width
                bottom_right_y = top_left_y + self.cell_height

                # Set background color: light blue for `1`, white for `0`
                cell_color = "lightblue" if value == 1 else "white"

                # Draw the matrix cell
                frame_draw.rectangle(
                    [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)],
                    outline="black",
                    fill=cell_color,
                )

                # Draw the value in the center of the cell
                # q_value = q_table.get_best_action(Position(r, c))
                # text_x = (top_left_x + bottom_right_x) // 2
                # text_y = (top_left_y + bottom_right_y) // 2
                # frame_draw.text(
                #     (text_x, text_y),
                #     str(q_value.value),
                #     fill="black",
                #     font=self.font,
                #     anchor="mm",  # Center the text
                # )

                # Measure text width for alignment
                text_sizes = {action_value.action: self.font.getbbox(str(action_value.value)) for action_value in q_table.get_cell(Position(r, c)).action_values}

                # Coordinates for each number
                text_positions = {
                    Action.UP: (top_left_x + self.cell_width // 2, top_left_y + 15),
                    Action.LEFT: (top_left_x + 30, top_left_y + self.cell_height // 2),
                    Action.RIGHT: (bottom_right_x - 30 - text_sizes[Action.RIGHT][0], top_left_y + self.cell_height // 2),
                    Action.DOWN: (top_left_x + self.cell_width // 2, bottom_right_y - 12),
                }
                # Draw each number
                for _action, text_position in text_positions.items():
                    q_action_value = q_table.get_action_value(Position(r, c), _action)
                    if prev_position is not None and r == prev_position.row and c == prev_position.col and _action == action:
                        color = "red"
                    else:
                        color = "black"
                    frame_draw.text(
                        text_position,
                        "{:.2f}".format(q_action_value.value),
                        fill=color,
                        font=self.font,
                        anchor="mm",  # Center align the text
                    )

        # Add additional information below the grid and matrix
        info_text_x = 10
        info_text_y = self.grid_height + 10  # Start below the grid and matrix

        # Draw current step and person's position
        frame_draw.text(
            (info_text_x, info_text_y),
            f"Episode: {episode}     Step: {step}     Epsilon: {"{:.3f}".format(epsilon)}     Random action: {is_action_random}",
            fill="black",
            font=self.font,
        )
        if action is not None:
            frame_draw.text(
                (info_text_x, info_text_y + 40),  # Next line
                f"new action value = {"{:.3f}".format(orig_current_action_value)} + {"{:.2f}".format(learning_rate)} * ({"{:.3f}".format(reward)} + {"{:.2f}".format(discount_rate)} * {"{:.3f}".format(new_best_action_value)} - {"{:.3f}".format(orig_current_action_value)}) = {"{:.3f}".format(new_current_action_value)}",
                fill="black",
                font=self.font,
            )

        # current_action_value.value = current_action_value.value +
        #                               lambda_learning_rate * (
        #                                           reward + gama_discount_rate * new_best_action_value.value - current_action_value.value)

        # Convert frame to OpenCV format
        frame_array = np.array(frame_image)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        self.video.write(frame_bgr)

    def close(self):
        self.video.release()
