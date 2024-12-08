package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.param.ParamMap;

import java.io.IOException;
import java.util.Arrays;

/**
 * Wine Quality Training using Spark ML
 * This class implements wine quality training using both Logistic Regression
 * and Random Forest Classification models.
 */
public class WineQualityTraining {
    // Constants for model configuration
    private static final String MODEL_SAVE_PATH = "s3://wine-quality-prediction/best_model";
    private static final String[] FEATURE_COLUMNS = {
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", 
        "pH", "sulphates", "alcohol", "quality"
    };

    public static void main(String[] args) throws IOException {
        // Initialize Spark session
        SparkSession spark = SparkSession.builder()
            .appName("Wine Quality Prediction")
            .getOrCreate();

        try {
            // Load and process training and validation datasets
            Dataset<Row> trainData = loadAndProcessData(spark, "s3://wine-quality-prediction/TrainingDataset.csv");
            Dataset<Row> validationData = loadAndProcessData(spark, "s3://wine-quality-prediction/ValidationDataset.csv");

            // Train and evaluate Logistic Regression model
            evaluateLogisticRegression(trainData, validationData);

            // Train and evaluate Random Forest model with cross-validation
            trainRandomForestWithCV(trainData);
        } finally {
            spark.stop();
        }
    }

    /**
     * Evaluates the Logistic Regression model performance
     */
    private static void evaluateLogisticRegression(Dataset<Row> trainData, Dataset<Row> validationData) {
        // Configure and train Logistic Regression model
        LogisticRegression lr = new LogisticRegression()
            .setLabelCol("quality")
            .setFeaturesCol("scaledFeatures")
            .setMaxIter(10)
            .setRegParam(0.3);
        
        LogisticRegressionModel lrModel = lr.fit(trainData);

        // Evaluate model performance
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("quality")
            .setPredictionCol("prediction");

        double trainAccuracy = evaluator.evaluate(lrModel.transform(trainData));
        double validationAccuracy = evaluator.evaluate(lrModel.transform(validationData));

        System.out.println("Logistic Regression Results:");
        System.out.println("Train Accuracy: " + trainAccuracy);
        System.out.println("Validation Accuracy: " + validationAccuracy);
    }

    /**
     * Trains Random Forest model using cross-validation and saves the best model
     */
    private static void trainRandomForestWithCV(Dataset<Row> trainData) throws IOException {
        RandomForestClassifier rf = new RandomForestClassifier()
            .setLabelCol("quality")
            .setFeaturesCol("scaledFeatures");

        // Configure hyperparameter grid for cross-validation
        ParamMap[] paramGrid = new ParamGridBuilder()
            .addGrid(rf.numTrees(), new int[] { 20, 50, 100 })
            .addGrid(rf.maxDepth(), new int[] { 5, 10, 15 })
            .build();

        // Setup cross-validation
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("quality")
            .setPredictionCol("prediction");

        CrossValidator cv = new CrossValidator()
            .setEstimator(rf)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3);

        // Train and evaluate model
        CrossValidatorModel cvModel = cv.fit(trainData);
        double bestScore = cvModel.avgMetrics()[0];

        System.out.println("\nRandom Forest Results:");
        System.out.println("Best Validation F1 Score: " + bestScore);

        // Save the best model
        RandomForestClassificationModel bestRfModel = (RandomForestClassificationModel) cvModel.bestModel();
        bestRfModel.write().overwrite().save(MODEL_SAVE_PATH);
        System.out.println("Best model saved to " + MODEL_SAVE_PATH);
    }

    private static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        Dataset<Row> df = spark.read().option("header", "true").option("sep", ";").option("inferSchema", "true")
                .csv(filePath);

        String[] columns = new String[] { "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                "quality" };
        df = df.toDF(columns);

        VectorAssembler assembler = new VectorAssembler().setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");
        StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
                .setWithStd(true).setWithMean(true);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { assembler, scaler });
        PipelineModel pipelineModel = pipeline.fit(df);
        return pipelineModel.transform(df);
    }
}
