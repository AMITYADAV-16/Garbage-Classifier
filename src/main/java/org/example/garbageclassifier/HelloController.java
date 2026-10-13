package org.example.garbageclassifier;

import com.github.sarxos.webcam.Webcam;
import ai.djl.modality.Classifications;
import javafx.animation.FadeTransition;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.util.Duration;
import javafx.scene.control.ComboBox;
import java.awt.image.BufferedImage;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.ResourceBundle;
import java.io.*;
import java.util.Properties;

public class HelloController implements Initializable {

    private Properties memory = new Properties();
    private final String MEMORY_FILE = "ai_memory.properties";

    private static class GarbageInfo {
        String binInfo;
        String usesInfo;

        GarbageInfo(String binInfo, String usesInfo) {
            this.binInfo = binInfo;
            this.usesInfo = usesInfo;
        }
    }

    private final Map<String, GarbageInfo> garbageDatabase = new HashMap<>();

    @FXML private ImageView cameraView;
    @FXML private Label garbageTypeLabel;
    @FXML private Label predictionsLabel;
    @FXML private Label binInfoLabel;
    @FXML private Label useInfoLabel;
    @FXML private Button scanButton;
    @FXML private ComboBox<String> correctionComboBox;

    private Webcam webcam;
    private boolean isCameraActive = false;
    private Thread cameraThread;
    private GarbageClassifier classifier;

    // All 10 classes matching the new dataset folder names exactly
    private static final String[] ALL_CLASSES = {
            "battery", "biological", "cardboard", "clothes",
            "glass", "metal", "paper", "plastic", "shoes", "trash"
    };

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        populateDatabase();

        // Populate dropdown with all 10 classes
        correctionComboBox.getItems().addAll(ALL_CLASSES);

        // Load existing corrections from disk
        try (InputStream input = new FileInputStream(MEMORY_FILE)) {
            memory.load(input);
        } catch (IOException ex) {
            System.out.println("No existing memory found. Starting fresh.");
        }

        scanButton.setDisable(true);
        scanButton.setText("System Loading...");
        garbageTypeLabel.setText("Initializing AI Engine...");
        garbageTypeLabel.setStyle("-fx-text-fill: #7f8c8d;");

        new Thread(this::initializeBackend).start();
    }

    private void initializeBackend() {
        webcam = Webcam.getWebcamByName("DroidCam Source 3 2");
        if (webcam == null) webcam = Webcam.getDefault();

        if (webcam != null) {
            webcam.open();
            isCameraActive = true;
            startCameraStream();
        } else {
            Platform.runLater(() -> garbageTypeLabel.setText("ERROR: No webcam found!"));
            return;
        }

        this.classifier = new GarbageClassifier();

        Platform.runLater(() -> {
            scanButton.setDisable(false);
            scanButton.setText("START SCAN");
            scanButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; -fx-background-radius: 20;");
            garbageTypeLabel.setText("System Ready");
        });
    }

    private void populateDatabase() {
        // 6 original recyclable classes
        garbageDatabase.put("cardboard",   new GarbageInfo("Bin: BLUE (Paper)",        "Uses: Recycled into new cardboard boxes."));
        garbageDatabase.put("glass",       new GarbageInfo("Bin: GREEN (Glass)",        "Uses: Recycled into new glass bottles."));
        garbageDatabase.put("metal",       new GarbageInfo("Bin: YELLOW (Recycling)",   "Uses: Recycled into new metal products."));
        garbageDatabase.put("paper",       new GarbageInfo("Bin: BLUE (Paper)",         "Uses: Recycled into new paper products."));
        garbageDatabase.put("plastic",     new GarbageInfo("Bin: YELLOW (Recycling)",   "Uses: Recycled into new plastic products."));
        garbageDatabase.put("trash",       new GarbageInfo("Bin: BLACK (General Waste)","Uses: Sent to energy-from-waste plants."));

        // 4 new classes
        garbageDatabase.put("battery",     new GarbageInfo("Bin: RED (Hazardous)",      "Uses: Take to a battery recycling point. Do NOT put in regular bins."));
        garbageDatabase.put("biological",  new GarbageInfo("Bin: BROWN (Organic)",      "Uses: Composted into fertilizer or biogas."));
        garbageDatabase.put("clothes",     new GarbageInfo("Bin: DONATION BOX",         "Uses: Donate if wearable, else textile recycling bank."));
        garbageDatabase.put("shoes",       new GarbageInfo("Bin: DONATION BOX",         "Uses: Donate if wearable, else take to shoe recycling point."));

        garbageDatabase.put("default",     new GarbageInfo("Bin: UNKNOWN",              "Uses: Check local guidelines for this item."));
    }

    @FXML
    protected void onScanButtonClick() {
        playScanEffect();
        scanButton.setDisable(true);
        scanButton.setText("ANALYZING...");
        garbageTypeLabel.setText("Processing...");

        BufferedImage currentImage = webcam.getImage();

        if (currentImage != null && classifier != null) {
            new Thread(() -> {
                String rawPrediction = classifier.predict(currentImage);
                String predictionKey = rawPrediction.toLowerCase().trim();

                // Memory check
                String rememberedSelection = memory.getProperty(predictionKey);
                boolean isFromMemory = false;
                if (rememberedSelection != null) {
                    predictionKey = rememberedSelection;
                    isFromMemory = true;
                }

                Classifications classifications = classifier.predictAll(currentImage);
                final String topKResults = formatTopK(classifications);
                final String finalKey = predictionKey;
                final boolean wasRemembered = isFromMemory;

                GarbageInfo info = garbageDatabase.getOrDefault(finalKey, garbageDatabase.get("default"));

                Platform.runLater(() -> {
                    scanButton.setDisable(false);
                    scanButton.setText("SCAN AGAIN");

                    if (classifications == null || classifications.topK(1).isEmpty()) {
                        showErrorState("No data returned from AI.");
                        return;
                    }

                    double topConfidence = classifications.topK(1).get(0).getProbability();

                    if (topConfidence < 0.5 && !wasRemembered) {
                        showErrorState("UNSURE / UNKNOWN");
                        useInfoLabel.setText("Low confidence (" + String.format("%.1f%%", topConfidence * 100) + ").");
                    } else {
                        String labelSuffix = wasRemembered ? " (FROM MEMORY)" : "";
                        garbageTypeLabel.setText(finalKey.toUpperCase().replace("_", " ") + labelSuffix);
                        predictionsLabel.setText(wasRemembered ? "Confidence: Recall from history" : topKResults);
                        binInfoLabel.setText(info.binInfo);
                        useInfoLabel.setText(info.usesInfo);
                        applyDynamicStyle(info.binInfo, topConfidence);
                    }
                });
            }).start();
        }
    }

    @FXML
    protected void onCorrectButtonClick() {
        String userSelection = correctionComboBox.getValue();
        String aiGuess = garbageTypeLabel.getText().toLowerCase()
                .replace(" (from memory)", "")
                .replace(" (corrected)", "")
                .replace(" ", "_").trim();

        // Fix: convert display name back to dataset key (no underscores in new classes)
        aiGuess = aiGuess.replace("_", " ").trim();

        if (userSelection != null && !aiGuess.isEmpty()
                && !aiGuess.equals("waiting...")
                && !aiGuess.equals("system ready")
                && !aiGuess.equals("processing...")) {

            memory.setProperty(aiGuess, userSelection);

            try (OutputStream output = new FileOutputStream(MEMORY_FILE)) {
                memory.store(output, "User Correction Rules");
                System.out.println("Saved: AI thought '" + aiGuess + "' → actually '" + userSelection + "'");
            } catch (IOException io) {
                io.printStackTrace();
            }

            GarbageInfo correctInfo = garbageDatabase.getOrDefault(userSelection, garbageDatabase.get("default"));
            garbageTypeLabel.setText(userSelection.toUpperCase() + " (CORRECTED)");
            garbageTypeLabel.setStyle("-fx-text-fill: #e67e22; -fx-font-weight: bold;");
            binInfoLabel.setText(correctInfo.binInfo);
            useInfoLabel.setText(correctInfo.usesInfo);
            predictionsLabel.setText("Confidence: Saved to Memory");
            applyDynamicStyle(correctInfo.binInfo, 1.0);
        }
    }

    private void showErrorState(String message) {
        garbageTypeLabel.setText(message);
        garbageTypeLabel.setStyle("-fx-text-fill: #e74c3c; -fx-font-weight: bold;");
        binInfoLabel.setText("Bin: Manual Check Required");
        binInfoLabel.setStyle("-fx-background-color: #95a5a6; -fx-text-fill: white; -fx-padding: 10; -fx-background-radius: 8;");
        predictionsLabel.setText("");
    }

    private void applyDynamicStyle(String binInfo, double confidence) {
        String color = "#95a5a6";

        if      (binInfo.contains("BLUE"))     color = "#3498db";
        else if (binInfo.contains("GREEN"))    color = "#27ae60";
        else if (binInfo.contains("YELLOW"))   color = "#f1c40f";
        else if (binInfo.contains("BLACK"))    color = "#34495e";
        else if (binInfo.contains("RED"))      color = "#e74c3c";
        else if (binInfo.contains("BROWN"))    color = "#a04000";
        else if (binInfo.contains("DONATION")) color = "#8e44ad";

        garbageTypeLabel.setStyle("-fx-text-fill: " + color + "; -fx-font-weight: bold; -fx-font-size: 24px;");
        binInfoLabel.setStyle("-fx-background-color: " + color + "22; -fx-text-fill: " + color +
                "; -fx-padding: 10; -fx-background-radius: 8; -fx-font-weight: bold; -fx-font-size: 16px;");
    }

    private String formatTopK(Classifications classifications) {
        if (classifications == null) return "No data";
        return classifications.topK(3).stream()
                .map(c -> {
                    double prob = Math.max(0.0, Math.min(1.0, c.getProbability()));
                    String className = c.getClassName().split(",")[0].trim();
                    return String.format("%s (%.1f%%)", className, prob * 100);
                })
                .reduce((a, b) -> a + "\n" + b)
                .orElse("");
    }

    private void playScanEffect() {
        FadeTransition ft = new FadeTransition(Duration.millis(150), cameraView);
        ft.setFromValue(1.0);
        ft.setToValue(0.3);
        ft.setCycleCount(2);
        ft.setAutoReverse(true);
        ft.play();
    }

    private void startCameraStream() {
        cameraThread = new Thread(() -> {
            while (isCameraActive) {
                BufferedImage bImage = webcam.getImage();
                if (bImage != null) {
                    Image fxImage = SwingFXUtils.toFXImage(bImage, null);
                    Platform.runLater(() -> cameraView.setImage(fxImage));
                }
                try { Thread.sleep(33); } catch (InterruptedException e) { break; }
            }
        });
        cameraThread.setDaemon(true);
        cameraThread.start();
    }

    public void closeCamera() {
        isCameraActive = false;
        if (webcam != null && webcam.isOpen()) {
            webcam.close();
        }
    }
}