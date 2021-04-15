import cv2
import numpy as np
import matplotlib.pyplot as plt


def nuimage_trajectory_visualization(translations, key_index,
                                     # sample_token: str,
                                     # rotation_yaw: float = 0.0,
                                     # center_key_pose: bool = True
                                     ) -> None:
        """
        Render a plot of the trajectory for the clip surrounding the annotated keyframe.
        A red cross indicates the starting point, a green dot the ego pose of the annotated keyframe.
        :param sample_token: Sample token.
        :param rotation_yaw: Rotation of the ego vehicle in the plot.
            Set to None to use lat/lon coordinates.
            Set to 0 to point in the driving direction at the time of the keyframe.
            Set to any other value to rotate relative to the driving direction (in radians).
        :param center_key_pose: Whether to center the trajectory on the key pose.
        :param out_path: Optional path to save the rendered figure to disk.
            If a path is provided, the plot is not shown to the user.
        """
        # Get the translations or poses.
        # translations, key_index = self.get_trajectory(sample_token, rotation_yaw=rotation_yaw,
        #                                               center_key_pose=center_key_pose)

        # Render translations.
        plt.figure()
        plt.plot(translations[:, 0], translations[:, 1])
        plt.plot(translations[key_index, 0], translations[key_index, 1], 'go', markersize=10)  # Key image.
        plt.plot(translations[0, 0], translations[0, 1], 'rx', markersize=10)  # Start point.
        max_dist = translations - translations[key_index, :]
        max_dist = np.ceil(np.max(np.abs(max_dist)) * 1.05)  # Leave some margin.
        max_dist = np.maximum(10, max_dist)
        plt.xlim([translations[key_index, 0] - max_dist, translations[key_index, 0] + max_dist])
        plt.ylim([translations[key_index, 1] - max_dist, translations[key_index, 1] + max_dist])
        plt.xlabel('x in meters')
        plt.ylabel('y in meters')
        # plt.show()
        plt.draw()
        plt.pause(0.001)
        # input()

def transform_to_image_space(coords, img_size):
    WIDTH = 75
    HEIGHT = 75
    new_coords = []

    # receive coord as x,y
    for i in range(len(coords)):
        coords[i][0] += WIDTH/2
        coords[i][0] = (coords[i][0] / WIDTH) * img_size[0]
        coords[i][1] = HEIGHT/2 - coords[i][1]
        coords[i][1] = (coords[i][1] / HEIGHT) * img_size[1]
        new_coords.append((int(coords[i][0]), int(coords[i][1])))
    return new_coords


def visualize_traffic(trajectory):
    # trajectory assumed x, y where x is horizontal axis and y is vertical axis
    img = np.ones((800, 800, 3))*255
    positions = transform_to_image_space(trajectory.copy(), (800, 800))

    for i in range(1, len(positions)):
        img = cv2.line(img, positions[i], positions[i-1], (255, 0, 0), 3)

    nuimage_trajectory_visualization(trajectory, 0)

    cv2.imshow("", img)
    cv2.waitKey(0)


def visualize_traffic_neighbours(trajectory, map_size):
    # trajectory assumed x, y where x is horizontal axis and y is vertical axis
    img = np.ones((800, 800, 3))*255

    for i in range(map_size):
        filter = (trajectory[i]!= [-64, -64, -64, -64, -64]).all(axis=1).tolist()
        positions = transform_to_image_space(trajectory[i][filter], (800, 800))
        R, G, B = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        R, G, B = 255, 0, 0

        for i in range(1, len(positions)):
            img = cv2.line(img, positions[i], positions[i-1], (B, G, R), 3)
            img = cv2.circle(img, positions[i-1], 3, (0, 0, 200), -1)
            # nuimage_trajectory_visualization(trajectory, 0)

    cv2.imshow("", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    visualize_traffic([[0, 0], [5,10], [20, 20]])

