package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	// Load in a dataset, with headers. Header attributes will be stored.
	// Think of instances as a Data Frame structure in R or Pandas.
	// You can also create instances from scratch.
	rawData, err := base.ParseCSVToInstances("/Users/dylancorbus/go/src/ml-go-project/datasets/iris.csv", true)
	if err != nil {
		panic(err)
	}

	// Print a pleasant summary of your data.
	fmt.Println(rawData)

	//Initialises a new KNN classifier
	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.10)
	cls.Fit(trainData)

	//create an instance to manually add data too
	inst := base.NewDenseInstances()
	//need to set up the instance to match the data set
	//add the attributes
	attrSpecsUnordered := base.ResolveAllAttributes(testData)
	attrSpecs := make([]base.AttributeSpec, len(attrSpecsUnordered))
	for x, a := range attrSpecsUnordered {
		i := a.GetAttribute()
		b := inst.AddAttribute(i)
		attrSpecs[x] = b
	}
	//adds the class attribute, class is the category
	inst.AddClassAttribute(inst.AllAttributes()[4])
	//allocates space for another row
	inst.Extend(1)
	//sets value for the row and column
	inst.Set(attrSpecs[0], 0, attrSpecs[0].GetAttribute().GetSysValFromString("5.9"))
	inst.Set(attrSpecs[1], 0, attrSpecs[1].GetAttribute().GetSysValFromString("3.0"))
	inst.Set(attrSpecs[2], 0, attrSpecs[2].GetAttribute().GetSysValFromString("5.1"))
	inst.Set(attrSpecs[3], 0, attrSpecs[2].GetAttribute().GetSysValFromString("1.8"))

	//make predictions on the testdata
	predictions, predErr := cls.Predict(testData)
	if predErr != nil {
		panic(predErr)
	}
	fmt.Println(predictions)

	//print relevant info
	confusionMat, confErr := evaluation.GetConfusionMatrix(testData, predictions)
	if confErr != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", confErr.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

	//make prediction on the single instance
	predictionsNew, newPredErr := cls.Predict(inst)
	if newPredErr != nil {
		panic(newPredErr)
	}
	//should be orange
	fmt.Println("SINGLE PREDICTION ", predictionsNew)
	cls.Save("/Users/dylancorbus/go/src/ml-go-project/internal/app/ml-model/model.txt")
}
