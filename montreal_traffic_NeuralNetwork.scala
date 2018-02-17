
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.sql.DataFrame

/**
  * Created by taiwoadetiloye on 2017-12-15.
  */
object montreal_traffic_NeuralNetwork extends App {

  Logger.getLogger("org").setLevel(Level.OFF)


  val dir = "./src/main/resources/"

  val path = dir + "Trafficdatamontreal.csv"


  val spark = org.apache.spark.sql.SparkSession.builder
    .master("local")
    .appName("Spark CSV Reader")
    .getOrCreate;

  import spark.implicits._

  val df = spark.read
    .format("csv")
    .option("header", "true") //reading the headers
    .option("mode", "DROPMALFORMED")
    .load(path)



  case class MontrealTraffic(Day:Double, Length: Double, Time: String, vtminus15: Double, vt: Double, vhistt: Double, vhisttplus15: Double,vtplus15predict: Double)

  val data = df.map { a => MontrealTraffic(a(0).toString.toDouble,a(1).toString.toDouble, a(2).toString, a(3).toString.toDouble, a(4).toString.toDouble, a(5).toString.toDouble,a(6).toString.toDouble, a(7).toString.toDouble) }
    .toDF("Day","Length","Time", "vtminus15", "vt", "vhistt", "vhisttplus15", "vtplus15predict")



  trait DataProcessor {
    def mainDataProcessor():DataFrame
  }




  class trainDataProcessor(data:DataFrame ) extends DataProcessor {

    val features = Array("Day", "Length", "vtminus15", "vt","vhistt","vhisttplus15")

    def mainDataProcessor =
    {
      val assembler = new VectorAssembler().setInputCols(features).setOutputCol("featureVectors")


      assembler.transform(data)
    }
  }


  val dataProcessed = new trainDataProcessor(data)

  val dataModel = dataProcessed.mainDataProcessor.select("featureVectors", "vtplus15predict")

  dataModel.show(10)


  // Split the data into train and test
  val splits = dataModel.randomSplit(Array(0.7, 0.3), seed = 1234L)
  val train = splits(0)
  val test = splits(1)

 train.printSchema()

  val count = data.select("vtplus15predict").distinct().count()
  print("count: " + count)


  val layers = Array[Int](6 ,7, 6, 35) //vary layers accordingly for NN - single layer to multilayer

  // create the trainer and set its parameters
  val mlp = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setFeaturesCol("featureVectors")
    .setLabelCol("vtplus15predict")
    .setMaxIter(1500)


  // train the model
  val model = mlp .fit(train)

  // compute accuracy on the test set
  val predictions = model.transform(test)
  val predictionAndLabels = predictions.select("prediction", "vtplus15predict")


  // Select (prediction, true label) and compute test error.
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("vtplus15predict")
    .setPredictionCol("prediction")


  val evaluator1 = evaluator.setMetricName("accuracy")
  val evaluator2 = evaluator.setMetricName("weightedPrecision")
  val evaluator3 = evaluator.setMetricName("weightedRecall")
  val evaluator4 = evaluator.setMetricName("f1")


  val accuracy = evaluator1.evaluate(predictions)
  val precision = evaluator2.evaluate(predictions)
  val recall = evaluator3.evaluate(predictions)
  val f1 = evaluator4.evaluate(predictions)


  println("Accuracy = " + accuracy)
  println("Precision = " + precision)
  println("Recall = " + recall)
  println("F1 = " + f1)
  println(s"Test Error = ${1 - accuracy}")

  val multiClassEvaluator = new MulticlassClassificationEvaluator()
  val auroc = multiClassEvaluator.setLabelCol("vtplus15predict").setPredictionCol("prediction").evaluate(predictions)
  println(s"Area under ROC = $auroc")


  spark.stop()


}
