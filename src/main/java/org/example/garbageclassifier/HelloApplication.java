package org.example.garbageclassifier;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class HelloApplication extends Application {
    @Override

    public void start(Stage stage) throws IOException {
        // We need to get the loader so we can access the controller
        FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("hello-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 800, 700); // Increased window size
        stage.setTitle("Garbage Classifier");
        stage.setScene(scene);
        stage.show();

        // This is the crucial part for closing the camera
        stage.setOnCloseRequest(event -> {
            HelloController controller = fxmlLoader.getController();
            controller.closeCamera();
        });
    }
    public static void main(String[] args) {
        launch();
    }
} 