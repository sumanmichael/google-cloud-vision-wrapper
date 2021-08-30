import glob
import os

from PIL import ImageOps

from GCVUtils import *

set_config_file("gocr-creds.json")

IMAGE_LOCATION = "data"
DATASET_LOCATION = "data/dataset"

for img_path in glob.glob(os.path.join(IMAGE_LOCATION, "*.jpg")):
    print("STARTING:", img_path)
    filename = img_path.split(os.sep)[-1].split(".")[0]

    image = Image.open(img_path)

    # EXIF CORRECTION OF ORIENTATION
    image = ImageOps.exif_transpose(image)
    image.save(img_path)

    response = get_api_response(img_path)
    os.makedirs(os.path.join(DATASET_LOCATION, filename), exist_ok=True)
    save_annotation_file(os.path.join(DATASET_LOCATION, filename, f"{filename}_gcv_response.json"), response)

    concave_bounds = get_bounds_from_annotation_file(
        os.path.join(DATASET_LOCATION, filename, f"{filename}_gcv_response.json"), is_concave=True)

    rect_bounds = get_bounds_from_annotation_file(
        os.path.join(DATASET_LOCATION, filename, f"{filename}_gcv_response.json"), is_concave=False)

    with open(os.path.join(DATASET_LOCATION, filename, f"{filename}_concave_hulls.json"), "w",
              encoding="utf-8") as f:
        json.dump(concave_bounds, f, ensure_ascii=False)

    with open(os.path.join(DATASET_LOCATION, filename, f"{filename}_rect_bounds.json"), "w",
              encoding="utf-8") as f:
        json.dump(rect_bounds, f, ensure_ascii=False)

    with open(os.path.join(DATASET_LOCATION, filename, f"{filename}.txt"), "w", encoding="utf-8") as f:
        for i, (coords, rect) in enumerate(zip(concave_bounds, rect_bounds)):
            get_cropped_masked_image(image, coords["vertices"], rect["vertices"]).save(
                os.path.join(DATASET_LOCATION, filename, f'{filename}-l{i}.jpg'))
            f.write(f'{filename}-l{i}.jpg {rect["text"]}\n')

    image = draw_polygons(image, bounds=[bound["vertices"] for bound in concave_bounds])
    # image = draw_bounds_for_features(image, os.path.join(DATASET_LOCATION, filename, f"{filename}_gcv_response.json"),[FeatureType.WORD])
    image.save(os.path.join(DATASET_LOCATION, f'{filename}_lines_annotated.jpg'))
    # image.save(os.path.join(DATASET_LOCATION, f'{filename}_lines_annotated_rect.jpg'))
