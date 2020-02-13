import numpy as np
import cv2
import util.gaze


def preprocess_unityeyes_image(img, json_data, oh=90, ow=150, heatmap_h=45, heatmap_w=75):
    # Prepare to segment eye image
    ih, iw = img.shape[:2]
    ih_2, iw_2 = ih/2.0, iw/2.0

    def process_coords(coords_list):
        coords = [eval(l) for l in coords_list]
        return np.array([(x, ih-y, z) for (x, y, z) in coords])
    
    interior_landmarks = process_coords(json_data['interior_margin_2d'])
    caruncle_landmarks = process_coords(json_data['caruncle_2d'])
    iris_landmarks = process_coords(json_data['iris_2d'])

    left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
    right_corner = interior_landmarks[8, :2]
    eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
    eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                          np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

    scale = ow/eye_width
    original_eyeball_radius = 71.7593
    eyeball_radius = original_eyeball_radius * scale  # See: https://goo.gl/ZnXgDE
    radius = np.float32(eyeball_radius)

    transform = np.zeros((2, 3))
    transform[0, 2] = -eye_middle[0] * scale + 0.5 * ow
    transform[1, 2] = -eye_middle[1] * scale + 0.5 * oh
    transform[0, 0] = scale
    transform[1, 1] = scale
    
    transform_inv = np.zeros((2, 3))
    transform_inv[:, 2] = -transform[:, 2]
    transform_inv[0, 0] = 1/scale
    transform_inv[1, 1] = 1/scale
    
    # Apply transforms
    eye = cv2.warpAffine(img, transform, (ow, oh))

    # Normalize eye image
    eye = eye.astype(np.float32)
    eye *= 2.0 / 255.0
    eye -= 1.0

    # Gaze
    # Convert look vector to gaze direction in polar angles
    look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3]
    look_vec[0] = -look_vec[0]
    original_gaze = util.gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
    gaze = util.gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
    if gaze[1] > 0.0:
        gaze[1] = np.pi - gaze[1]
    elif gaze[1] < 0.0:
        gaze[1] = -(np.pi + gaze[1])
    gaze = gaze.astype(np.float32)

    iris_centre = np.asarray([
        iw_2 + original_eyeball_radius * -np.cos(original_gaze[0]) * np.sin(original_gaze[1]),
        ih_2 + original_eyeball_radius * -np.sin(original_gaze[0]),
    ])
    landmarks = np.concatenate([interior_landmarks[::2, :2],  # 8
                                iris_landmarks[::4, :2],  # 8
                                iris_centre.reshape((1, 2)),
                                [[iw_2, ih_2]],  # Eyeball centre
                                ])  # 18 in total
    landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1))
    landmarks = np.asarray(landmarks * transform.T)
    landmarks = landmarks.astype(np.float32)

    heatmaps = get_heatmaps((oh, ow), landmarks)
    heatmaps = np.array([cv2.resize(x, (heatmap_w, heatmap_h), interpolation=cv2.INTER_CUBIC) for x in heatmaps])

    return {
        'img': eye,
        'transform': transform,
        'transform_inv': transform_inv,
        'radius': radius,
        'original_radius': original_eyeball_radius,
        'eye_middle': eye_middle,
        'heatmaps': heatmaps,
        'landmarks': landmarks,
        'gaze': gaze
    }


def get_heatmaps(shape, landmarks):

    def gaussian_2d(shape, centre, sigma=1.0):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
        ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
        alpha = -0.5 / (sigma ** 2)
        heatmap = np.exp(alpha * ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2))
        return heatmap

    heatmaps = []
    for (x, y) in landmarks:
        heatmaps.append(gaussian_2d(shape, (int(x), int(y)), sigma=3.0))
    return heatmaps


# get heatmaps
def gaussian_2d(shape, centre, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
    ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - centre[1]) ** 2 + (ys - centre[0]) ** 2))
    return heatmap