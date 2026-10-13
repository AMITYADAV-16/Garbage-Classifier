package org.example.garbageclassifier;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;

public class GarbageClassifier {

    private Predictor<Image, Classifications> predictor;

    public GarbageClassifier() {
        try {

            ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                    .addTransform(a -> NDImageUtils.resize(a, 224, 224))
                    // REMOVED: .div(255.0f) <- ToTensor() does this automatically!

                    .addTransform(new ToTensor()) // This scales to [0,1] AND flips to CHW

                    .addTransform(new Normalize(
                            new float[] {0.485f, 0.456f, 0.406f},
                            new float[] {0.229f, 0.224f, 0.225f}))

                    .optApplySoftmax(true)

                    .build();

            Criteria<Image, Classifications> criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optModelPath(Paths.get("models"))
                    .optModelName("garbage_mobilenetv2")
                    .optEngine("PyTorch")
                    .optTranslator(translator)
                    .build();

            ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
            this.predictor = model.newPredictor();

        } catch (ModelException | IOException e) {
            e.printStackTrace();
        }
    }

    public String predict(BufferedImage bufferedImage) {
        if (predictor == null) return "default";
        try {
            Image img = ImageFactory.getInstance().fromImage(bufferedImage);
            Classifications classifications = predictor.predict(img);

            // 1. Get the raw best result
            Classifications.Classification best = classifications.best();
            String className = best.getClassName();

            // 2. SAFETY FILTER: If the model returns a raw number (like "7" or "8")
            // it means it's guessing a class that doesn't exist in your 6-line synset.txt
            if (className.matches("\\d+")) {
                System.out.println("AI guessed a hidden category index: " + className);
                return "waste"; // Default to waste or general 'default'
            }

            // 3. Normal path: clean the text and return
            return className.split(",")[0].trim().toLowerCase();

        } catch (Exception e) {
            // If the AI math fails or the index is out of bounds,
            // return 'default' so the UI button re-enables
            System.err.println("Prediction error caught: " + e.getMessage());
            return "default";
        }
    }
    public Classifications predictAll(BufferedImage bufferedImage) {
        if (predictor == null) return null;
        try {
            Image img = ImageFactory.getInstance().fromImage(bufferedImage);
            return predictor.predict(img);
        } catch (TranslateException e) {
            e.printStackTrace();
            return null;
        }
    }
}