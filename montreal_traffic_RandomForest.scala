import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{RFormula, SQLTransformer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

/**
  * Created by taiwoadetiloye on 2017-12-08.
  */
object montreal_traffic_RandomForest extends App {

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
    .load(path).cache()



  case class MontrealTraffic(Day:Double, Length: Double, Time: String, vtminus15: Double, vt: Double, vhistt: Double, vhisttplus15: Double,vtplus15predict: Double)

  val data = df.map { a => MontrealTraffic(a(0).toString.toDouble,a(1).toString.toDouble, a(2).toString, a(3).toString.toDouble, a(4).toString.toDouble, a(5).toString.toDouble,a(6).toString.toDouble, a(7).toString.toDouble) }
    .toDF("Day","Length","Time", "vtminus15", "vt", "vhistt", "vhisttplus15", "vtplus15predict")


  //data.printSchema()
  //data.show(10)

  /*
  *  This function helps to select Indexed columns obtained from the stringIndex function using SQLTransformer
  */

  trait DataProcessor {
    def mainDataProcessor():DataFrame
  }


  /* RFormula produces a vector column of features and a double or string column of label.
  Like when formulas are used in R for linear regression, string input columns will be one-hot encoded,
   and numeric columns will be cast to doubles. If the label column is of type string,
   it will be first transformed to double with StringIndexer.
   If the label column does not exist in the DataFrame, If the label column does not exist in the DataFrame,

   */

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




  // Split the data into train and test
  val splits = dataModel.randomSplit(Array(0.7, 0.3), seed = 1234L)
  val train = splits(0)
  val test = splits(1)



  // create the trainer and set its parameters
  val rf = new RandomForestClassifier()
    .setFeaturesCol("featureVectors")
    .setLabelCol("vtplus15predict")
    .setNumTrees(10)
//    .setSeed(1344L)
//    .setMaxDepth(8).setMaxBins(100)




  val pipeline = new Pipeline()
    .setStages(Array( rf))

  // train the model
  val model = pipeline.fit(train)

  // compute accuracy on the test set
  val predictions = model.transform(test)
  val predictionAndLabels = predictions.select("prediction", "vtplus15predict").show(10)


  // Select (prediction, true label) and compute test error.
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




  val rfModel = model.stages(0).asInstanceOf[RandomForestClassificationModel]
//println("Learned classification forest model:\n" + rfModel.toDebugString)

  val multiClassEvaluator = new MulticlassClassificationEvaluator()
  val auroc = multiClassEvaluator.setLabelCol("vtplus15predict").setPredictionCol("prediction").evaluate(predictions)
  println(s"Area under ROC = $auroc")

  spark.stop()


//  You should observe the value as follows:
//
//    Accuracy = 0.5217246545696688
//  Precision = 0.488360500637862
//  Recall = 0.5217246545696688
//  F1 = 0.4695649096879411
//  Test Error = 0.47827534543033123
}






