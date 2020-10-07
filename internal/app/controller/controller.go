package controller

import (
	"encoding/json"
	"fmt"
	"github.com/dylancorbus/ml-go-project/internal/app/models"
	"github.com/dylancorbus/ml-go-project/pkg/iris/service"
	"net/http"
)

var counter int

func Start() {
	http.HandleFunc("/predict", predict)
	http.HandleFunc("/hello", hello)
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("./web/static"))))
	http.ListenAndServe(":8080", nil)
}

func hello(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
}

func predict(w http.ResponseWriter, r *http.Request) {
	// Unmarshal
	var flower models.Flower
	if r.Body == nil {
		fmt.Println(w, "Please send a request body", 400)
		return
	}
	err := json.NewDecoder(r.Body).Decode(&flower)
	if err != nil {
		panic(err)
	}
	//log request
	fmt.Println("NEW REQUEST ", flower)

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Credentials", "true")
	w.Header().Set("Access-Control-Max-Age", "1800")
	w.Header().Set("Access-Control-Allow-Headers", "content-type")
	w.Header().Set("Access-Control-Allow-Methods","PUT, POST, GET, DELETE, PATCH, OPTIONS")

	//call iris
	prediction := service.Predict(flower)
	fmt.Fprint(w, prediction)
	counter++
	fmt.Println(counter)
}
