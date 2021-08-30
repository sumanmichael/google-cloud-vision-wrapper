import io
import json
import os
from enum import Enum

import alphashape
from PIL import Image, ImageDraw
from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def set_config_file(config_file_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config_file_path


def get_api_response(image_file):
    client = vision.ImageAnnotatorClient()

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    return response


def get_document_bounds(filename, feature):
    """Returns document bounds given an image."""

    bounds = []
    document = get_response_from_file(filename)["fullTextAnnotation"]

    # Collect specified feature bounds by enumerating all document features
    for page in document["pages"]:
        for block in page["blocks"]:
            for paragraph in block["paragraphs"]:
                for word in paragraph["words"]:
                    for symbol in word["symbols"]:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(symbol["boundingBox"])

                    if feature == FeatureType.WORD:
                        bounds.append(word["boundingBox"])

                if feature == FeatureType.PARA:
                    bounds.append(paragraph["boundingBox"])

            if feature == FeatureType.BLOCK:
                bounds.append(block["boundingBox"])

    # The list `bounds` contains the coordinates of the bounding boxes.
    bounds_points = []
    for bound in bounds:
        bounds_points.append([(v["x"], v["y"]) for v in bound["vertices"]])
    return bounds_points


def draw_polygons(image, bounds, color="red"):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        points = []
        for x, y in bound:
            points += [x, y]
        draw.polygon(points, None, color)
    return image


def draw_rectangles(image, edge_vertices, color="red"):
    draw = ImageDraw.Draw(image)

    for vertices in edge_vertices:
        draw.rectangle(vertices, outline=color)

    return image


def draw_bounds_for_features(image, filename, features):
    colors = ["blue", "green", "yellow", "red", "black"]
    for feature in features:
        bounds = get_document_bounds(filename, feature)
        draw_polygons(image, bounds, colors[feature.value - 1])

    return image


def save_annotation_file(filename, response):
    with open(filename, "w", encoding="utf-8") as f:
        response_json = AnnotateImageResponse.to_json(response)
        response_dict = json.loads(response_json)
        json.dump(response_dict, f, ensure_ascii=False)


def get_bounds_from_annotation_file(filename, is_concave=True, alpha_concave=None):
    annotation_dict = get_response_from_file(filename)
    annotation = annotation_dict["fullTextAnnotation"]
    lines = []

    breaks = vision.TextAnnotation.DetectedBreak.BreakType
    for page in annotation["pages"]:
        for block in page["blocks"]:
            for paragraph in block["paragraphs"]:
                line = ""
                word_bboxes = []
                for word in paragraph["words"]:
                    word_bboxes.append([(v["x"], v["y"]) for v in word["boundingBox"]["vertices"]])
                    for symbol in word["symbols"]:
                        line += symbol["text"]
                        try:
                            symbol_breakType = symbol['property']['detectedBreak']['type']
                            if symbol_breakType in map(lambda x: x.value,
                                                       [breaks.SPACE, breaks.EOL_SURE_SPACE, breaks.LINE_BREAK,
                                                        breaks.HYPHEN]):
                                if symbol_breakType in map(lambda x: x.value, [breaks.SPACE, breaks.EOL_SURE_SPACE]):
                                    line += ' '

                                if symbol_breakType in map(lambda x: x.value,
                                                           [breaks.EOL_SURE_SPACE, breaks.LINE_BREAK, breaks.HYPHEN]):
                                    if symbol_breakType == breaks.HYPHEN.value:
                                        line += "-"
                                    lines.append({
                                        "text": line,
                                        "vertices": concave_hull(word_bboxes, alpha_concave)
                                                        if is_concave
                                                        else merge_boxes(word_bboxes)
                                    })
                                    word_bboxes = []
                                    line = ''
                            else:
                                raise RuntimeWarning(
                                    f"BreakType - {breaks(symbol_breakType).__repr__()} is Not Handled")
                        except KeyError:
                            pass

    return lines


def get_response_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        annotation_dict = json.load(f)
    return annotation_dict


def concave_hull(boxes, alpha=None):
    all_boxes = sum(boxes, [])
    if alpha is not None:
        alpha_shape = alphashape.alphashape(all_boxes, alpha=alpha)
    else:
        alpha_shape = alphashape.alphashape(all_boxes)
    hull_pts = alpha_shape.exterior.coords.xy
    return [(x, y) for x, y in zip(*hull_pts)]



def merge_boxes(boxes):
    all_boxes = sum(boxes, [])
    minx = min(all_boxes, key=lambda x: x[0])[0]
    miny = min(all_boxes, key=lambda x: x[1])[1]
    maxx = max(all_boxes, key=lambda x: x[0])[0]
    maxy = max(all_boxes, key=lambda x: x[1])[1]

    return minx, miny, maxx, maxy


def get_cropped_masked_image(img, coords, rect_bound):
    # img = img.rotate(-90, PIL.Image.NEAREST, expand=1)
    img = img.crop(rect_bound)

    back = Image.new("L", img.size, 0)
    draw_mask = ImageDraw.Draw(back)

    draw_mask.polygon([(x - rect_bound[0], y - rect_bound[1]) for x, y in coords], fill=255)
    img = Image.composite(img, back, back)
    return img
