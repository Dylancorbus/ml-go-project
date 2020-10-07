package service

import (
	"fmt"
	"github.com/dylancorbus/ml-go-project/internal/app/models"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
)

var (
	cls *knn.KNNClassifier
	clsErr error
	rawData *base.DenseInstances
	dataErr error
	err error
)

func init() {
	cls, clsErr = knn.ReloadKNNClassifier("/Users/dylancorbus/go/src/ml-go-project/internal/app/ml-model/model.txt")
	if clsErr != nil {
		panic(clsErr)
	}
	rawData, dataErr = base.ParseCSVToInstances("/Users/dylancorbus/go/src/ml-go-project/datasets/iris.csv", true)
	if dataErr != nil {
		panic(dataErr)
	}

}

func Predict(flower models.Flower) base.FixedDataGrid {
	//adds the class attribute, class is the category
	attrSpecsUnordered := base.ResolveAllAttributes(rawData)
	attrSpecs := make([]base.AttributeSpec, len(attrSpecsUnordered))

	inst := base.NewDenseInstances()
	for x, a := range attrSpecsUnordered {
		i := a.GetAttribute()
		b := inst.AddAttribute(i)
		attrSpecs[x] = b
	}
	//adds the class attribute, class is the category
	inst.AddClassAttribute(inst.AllAttributes()[4])
	//allocates space for another row
	inst.Extend(1)

	inst.Set(attrSpecs[0], 0, attrSpecs[0].GetAttribute().GetSysValFromString(fmt.Sprintf("%f", flower.Height)))
	inst.Set(attrSpecs[1], 0, attrSpecs[1].GetAttribute().GetSysValFromString(fmt.Sprintf("%f", flower.Width)))
	inst.Set(attrSpecs[2], 0, attrSpecs[2].GetAttribute().GetSysValFromString(fmt.Sprintf("%f", flower.Weight)))
	inst.Set(attrSpecs[3], 0, attrSpecs[3].GetAttribute().GetSysValFromString(fmt.Sprintf("%f", flower.Water)))

	predictionsNew, newPredErr := cls.Predict(inst)
	if newPredErr != nil {
		panic(newPredErr)
	}

	fmt.Println("SINGLE PREDICTION ", predictionsNew)
	return predictionsNew
}