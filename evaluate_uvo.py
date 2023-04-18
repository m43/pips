import argparse

from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_json', type=str,
                        default='./data/UVOv1.0/VideoDenseSet/UVO_video_val_dense.json')
    parser.add_argument('--predictions_json', type=str,
                        default='./data/UVOv1.0/ExampleSubmission/video_val_pred.json')
    return parser


def evaluate(test_annotation_file, user_submission_file):
    """
    Evaluates the submission and returns score Arguments:

        `test_annotation_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
    """
    print("Starting Evaluation.....")
    uvo_api = YTVOS(test_annotation_file)
    print(user_submission_file)
    uvo_det = uvo_api.loadRes(user_submission_file)
    # convert ann in uvo_det to class-agnostic
    for ann in uvo_det.dataset["annotations"]:
        if ann["category_id"] != 1:
            ann["category_id"] = 1

    # start evaluation
    uvo_eval = YTVOSeval(uvo_api, uvo_det, "segm")
    uvo_eval.params.useCats = False
    uvo_eval.evaluate()
    uvo_eval.accumulate()
    uvo_eval.summarize()

    output = {}
    output["result"] = [
        {
            "UVO frame results": {
                "AR@100": uvo_eval.stats[8],
                "AP": uvo_eval.stats[0],
                "AP.5": uvo_eval.stats[1],
                "AP.75": uvo_eval.stats[2],
                "AR@1": uvo_eval.stats[6],
                "AR@10": uvo_eval.stats[7],
            }
        },
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]
    return output


if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.annotations_json, args.predictions_json)
