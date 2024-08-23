using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data;

public class StormData
{
    // Removed the Date field from the class since it's no longer in the data file
    [LoadColumn(0)]
    public float Latitude { get; set; }

    [LoadColumn(1)]
    public float Longitude { get; set; }

    [LoadColumn(2)]
    public float WindSpeed { get; set; }

    [LoadColumn(3)]
    public float Pressure { get; set; }
}
public class LatitudePrediction
{
    [ColumnName("Score")]
    public float PredictedLatitude { get; set; }
}

public class LongitudePrediction
{
    [ColumnName("Score")]
    public float PredictedLongitude { get; set; }
}


class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load data from TSV file
        IDataView dataView = mlContext.Data.LoadFromTextFile<StormData>(
            @"C:\GenCodeMain\MLTest-101\SampleData\fakestorm.tsv",
            hasHeader: true,
            separatorChar: '\t'
        );

        // Train model for Latitude
        var pipelineLat = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(StormData.Latitude))
            .Append(mlContext.Transforms.Concatenate("Features", "Longitude", "WindSpeed", "Pressure"))
            .Append(mlContext.Regression.Trainers.FastTree());

        var modelLat = pipelineLat.Fit(dataView);

        // Train model for Longitude
        var pipelineLon = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(StormData.Longitude))
            .Append(mlContext.Transforms.Concatenate("Features", "Latitude", "WindSpeed", "Pressure"))
            .Append(mlContext.Regression.Trainers.FastTree());

        var modelLon = pipelineLon.Fit(dataView);

        // Create prediction engines
        var predEngineLat = mlContext.Model.CreatePredictionEngine<StormData, LatitudePrediction>(modelLat);
        var predEngineLon = mlContext.Model.CreatePredictionEngine<StormData, LongitudePrediction>(modelLon);

        // Display results together
        Console.WriteLine("Predicted Coordinates:");
        var predictions = mlContext.Data.CreateEnumerable<StormData>(dataView, reuseRowObject: false);
        foreach (var stormData in predictions)
        {
            var predictionLat = predEngineLat.Predict(stormData);
            var predictionLon = predEngineLon.Predict(stormData);

            Console.WriteLine($"Latitude: {predictionLat.PredictedLatitude}, Longitude: {predictionLon.PredictedLongitude}");
        }

        // Save models
        mlContext.Model.Save(modelLat, dataView.Schema, @"C:\GenCodeMain\MLTest-101\SampleData\latmodel.zip");
        mlContext.Model.Save(modelLon, dataView.Schema, @"C:\GenCodeMain\MLTest-101\SampleData\lonmodel.zip");

        GetPredictedNextCoordinate();
    }

    public static void GetPredictedNextCoordinate()
    {
        var mlContext = new MLContext();

        // Load models
        ITransformer latModel = mlContext.Model.Load("C:\\GenCodeMain\\MLTest-101\\SampleData\\latmodel.zip", out var latSchema);
        ITransformer lonModel = mlContext.Model.Load("C:\\GenCodeMain\\MLTest-101\\SampleData\\lonmodel.zip", out var lonSchema);

        // Prediction engines
        var latPredEngine = mlContext.Model.CreatePredictionEngine<StormData, LatitudePrediction>(latModel);
        var lonPredEngine = mlContext.Model.CreatePredictionEngine<StormData, LongitudePrediction>(lonModel);

        // Example data (last known data point)
        var stormData = new StormData
        {
            Latitude = 25.0f, // Last coordinate for Andrew in my tsv file
            Longitude = -80.0f,
            WindSpeed = 150f,
            Pressure = 922f
        };

        // Predict the next coordinates
        var latPrediction = latPredEngine.Predict(stormData);
        var lonPrediction = lonPredEngine.Predict(stormData);

        // Print predictions
        Console.WriteLine($"Next Predicted Latitude: {latPrediction.PredictedLatitude}");
        Console.WriteLine($"Next Predicted Longitude: {lonPrediction.PredictedLongitude}");
    }

}
