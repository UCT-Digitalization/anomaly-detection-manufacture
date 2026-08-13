## PatchCore Manufacturing Installation Anomaly Detection Proof-Of-Concept

Problem Statement: For inspection of machineries / mechanical onjects, there may be too many points to inspect everytime. This may result to operators' fatigue, leading to inaccurate inspections.

Idea: Train an deep learning model to detect anomalies from normal conditions (golden images training data).
Patchcore trains on ONLY golden images to form an embedded memory bank. During inference, the model will calculate the similarity distance between the input image with the memory bank. For input images that are similar/close to golden images in the memory bank, the distance score will be very low (closer to 0 the better). For input image that are very different fromt he golden images, the distance score will be much larger (higher the better, i.e. anomalies.)

For more detailed information on the PatchCore technical specifications, refer to [[PatchCore Manufacturing Specs.md]](/PatchCore%20Manufacturing%20Specs.md).

#### Files directory

[[data]](/data/) - step_01 refer to a single inspection point/location. So for more inspection points, there should be step_02, 03, etc. Inside each step, there should be test and train folder. Train folder contains good folder images, while test folder contains good and bad folders' images.

[[results]](/results/Patchcore/) - contains trained models' output results in images along with the weights saved. The library will automatically create v0, v1, v2, etc when u retrain.

[[convert_heic_jpg.py]](/convert_heic_jpg.py) - script to convert .HEIC images to .JPEG format, if iphone is being used

[[inspection_dataset.py]](/inspect_dataset.py) - check for any non JPG format images left

[[train.py]](/train.py) - script to train, validate and test a trained model. For validation and testing, PatchCore by default will split 50% of images in test folder for validation to fine tune model, the other 50% for testing to calculate scores.  Run `python train.py` to start the training. You can variate the image_size and feature_layers to see how it affects the model training results.

[[individual_test.py]](/individual_test.py) - script to test ALL good and bad images and calculate the anomalies, accuracy, BAD detection rate and good acceptance rate. The threshold to determine good / bad images by default is calculated by the model by performing optimal calculation: Elbow method of separation. If you want to adjust the threshold manually, e.g. to improve recall (catch more anomalies instead of good images), get their actual score calculation with `get_scalar(scores[i])`, from `scores = batch.pred_score`. Example below:

```python
for batch in predictions:

    paths = batch.image_path
    scores = batch.pred_score
    labels = batch.pred_label

    for i, path in enumerate(paths):

        path = Path(path)

        score = get_scalar(scores[i])

        anomalib_label = get_scalar(labels[i])

        actual_label = (
            "GOOD"
            if path.parent.name.lower() == "good"
            else "BAD"
        )

        anomalib_predicted = (
            "ANOMALY"
            if int(anomalib_label) == 1
            else "NORMAL"
        )

        predicted = (
            "ANOMALY"
            if score >= THRESHOLD
            else "NORMAL"
        )

        expected = (
            "NORMAL"
            if actual_label == "GOOD"
            else "ANOMALY"
        )
```
-------------------------------------------------------------------------------------------

### Model Evaluation Summary

| Version | Image Size | Backbone | Feature Layers | Coreset Sampling Ratio | Images Tested (Good / Bad) | Total test Images | Accuracy | BAD Detection Rate | GOOD Acceptance Rate | Remarks |
|----------|------------|-----------|----------------|------------------------|----------------------------|--------------|----------|-------------------|---------------------|---------|
| v0 | 512 × 512 | wide_resnet101_2 | layer 2, layer 3 | 0.1 | 12 / 12 | 24 | 95.83% | 91.67% | 100.00% | Baseline evaluation |
| v1 | 512 × 512 | wide_resnet101_2 | layer 2, layer 3 | 0.1 | 19 / 25 | 44 | 86.36% | 88.00% | 84.21% | Expanded test set |
| v2 | 1024 × 1024 | wide_resnet101_2 | layer 2, layer 3 | 0.1 | 19 / 25 | 44 | 86.36% | 84.00% | 89.47% | Increased image size |
| v3 | 512 × 512 | wide_resnet101_2 | layer 1, layer 2 | 0.1 | 19 / 25 | 44 | 90.91% | 88.00% | 94.74% | Changed feature layers |
| v4 | 512 × 512 | wide_resnet101_2 | layer 2, layer 3 | 0.1 | 42 / 68 | 110 | 91.82% | 92.65% | 90.48% | Larger training & test dataset |
| **v4.5** | **512 × 512** | **wide_resnet101_2** | **layer 2, layer 3** | **0.1** | **42 / 68** | **110** | **93.64%** | **98.53%** | **85.71%** | **Threshold adjusted lower (0.50 → 0.46)** |

#### Key Observations

- **v0** achieved the highest GOOD Acceptance Rate (**100%**), but was evaluated on a relatively small dataset.
- **v3** provided the best balance between BAD Detection Rate and GOOD Acceptance Rate on the 44-image test set.
- **v4** improved overall robustness by using a larger training and testing dataset.
- **v4.5** achieved the highest BAD Detection Rate (**98.53%**) and overall Accuracy (**93.64%**) by lowering the anomaly threshold from **0.50** to **0.46**.
- The threshold adjustment significantly reduced false accepts (defective images classified as normal) but increased false rejects (good images classified as anomalies).

#### Recommended Model

> **v4.5** is recommended for production defect screening where detecting defective parts is prioritized over minimizing false alarms. It provides the highest BAD Detection Rate (98.53%) while maintaining a strong overall Accuracy (93.64%).

#### Demo results

##### Image with detected anomaly 

assets\anomaly image output.jpg

##### Image with correct/normal condition

assets\good image output.jpg