import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from task_benchmark import evaluate


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0bIDATx\x9cc`\x00\x02\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


if __name__ == "__main__":
	with TemporaryDirectory() as tmp_dir:
		tmp_path = Path(tmp_dir)
		inputs_dir = tmp_path / "images"
		inputs_dir.mkdir(parents=True, exist_ok=True)

		(inputs_dir / "img1.bin").write_bytes(MINIMAL_PNG)
		(inputs_dir / "img2.bin").write_bytes(MINIMAL_PNG)

		dataset_path = tmp_path / "dataset.csv"
		with dataset_path.open("w", newline="") as fh:
			writer = csv.DictWriter(
				fh,
				fieldnames=["image_path", "wnid", "class_description"],
			)
			writer.writeheader()
			writer.writerow(
				{
					"image_path": "img1.bin",
					"wnid": "n01440764",
					"class_description": "tench",
				}
			)
			writer.writerow(
				{
					"image_path": "img2.bin",
					"wnid": "n01443537",
					"class_description": "tench",
				}
			)

		report = evaluate(
			dataset_path=dataset_path,
			task="image-classification",
			implementation="most-common-class",
			task_inputs_dir_path=inputs_dir,
			implementation_import_path="my_custom_model.most_common",
			device="cpu",
			batch_size=2,
		)

		print("Top-1 accuracy:", report["top1_accuracy"])
		print("Top-5 accuracy:", report["top5_accuracy"])
