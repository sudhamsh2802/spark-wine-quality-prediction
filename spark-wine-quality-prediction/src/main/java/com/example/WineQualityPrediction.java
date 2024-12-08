package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.util.Arrays;

/**
 * Wine Quality Prediction Model Inference
 * This class loads a pre-trained Random Forest model and performs predictions
 * on validation data.
 */
public class WineQualityPrediction {
    // Constants for default paths and feature configuration
    private static final String DEFAULT_MODEL_PATH = "s3://wine-quality-prediction/best_model/";
    private static final String DEFAULT_TEST_PATH = "s3://wine-quality-prediction/ValidationDataset.csv";
    private static final String[] FEATURE_COLUMNS = {
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol", "quality"
    };

    public static void main(String[] args) {
        // Process command-line arguments with defaults
        String modelPath = args.length > 0 ? args[0] : DEFAULT_MODEL_PATH;
        String testFilePath = args.length > 1 ? args[1] : DEFAULT_TEST_PATH;

        // Initialize Spark Session
        SparkSession spark = SparkSession.builder()
            .appName("Wine Quality Prediction")
            .getOrCreate();

        try {
            // Print environment details for debugging
            printSparkEnvironmentDetails(spark);

            // Load and process validation data
            Dataset<Row> validationData = loadAndProcessData(spark, testFilePath);
            System.out.println("\nScaled Features Preview:");
            validationData.select("scaledFeatures").show(false);

            // Perform model inference and evaluation
            evaluateModel(validationData, modelPath);
        } finally {
            spark.stop();
        }
    }

    /**
     * Prints Spark environment configuration details
     */
    private static void printSparkEnvironmentDetails(SparkSession spark) {
        System.out.println("Spark Configuration:");
        System.out.println("- Version: " + spark.version());
        System.out.println("- App Name: " + spark.sparkContext().appName());
        System.out.println("- Master: " + spark.sparkContext().master());
        System.out.println("- Application Id: " + spark.sparkContext().applicationId());
    }

    /**
     * Loads the model and evaluates its performance on the validation dataset
     */
    private static void evaluateModel(Dataset<Row> validationData, String modelPath) {
        // Load the saved model
        RandomForestClassificationModel loadedModel = RandomForestClassificationModel.load(modelPath);
        
        // Make predictions
        Dataset<Row> predictions = loadedModel.transform(validationData);
        System.out.println("\nPredictions Preview:");
        predictions.show();

        // Evaluate model performance
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("quality")
            .setPredictionCol("prediction")
            .setMetricName("f1");
            
        double score = evaluator.evaluate(predictions);
        System.out.println("\nModel Performance:");
        System.out.println("F1 Score = " + score);
    }

    /**
     * Loads and preprocesses the input data
     */
    private static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        // Load CSV data
        Dataset<Row> df = spark.read()
            .option("header", "true")
            .option("sep", ";")
            .option("inferSchema", "true")
            .csv(filePath);

        // Rename columns to match expected schema
        df = df.toDF(FEATURE_COLUMNS);

        // Configure feature processing pipeline
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(Arrays.copyOfRange(FEATURE_COLUMNS, 0, FEATURE_COLUMNS.length - 1))
            .setOutputCol("features");

        StandardScaler scaler = new StandardScaler()
            .setInputCol("features")
            .setOutputCol("scaledFeatures")
            .setWithStd(true)
            .setWithMean(true);

        // Create and apply the pipeline
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, scaler});
        PipelineModel pipelineModel = pipeline.fit(df);
        return pipelineModel.transform(df);
    }
}
