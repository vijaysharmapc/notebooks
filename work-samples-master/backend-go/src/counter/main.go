package main

import (
	"database/sql"
	"fmt"
	"net/http"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

var pageCount = map[string]int64{"view": 0}
var pageCountLock = sync.Mutex{}
var db *sql.DB
var dbLock = sync.Mutex{}

func welcomeHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "Welcome to EQ Works 😎")
}

func viewHandler(w http.ResponseWriter, r *http.Request) {
	incCounter("view")
	processRequest(r.RequestURI)
	fmt.Fprint(w, "success")
}

func counterHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "%v", pageCount)
}

func counterDBHandler(w http.ResponseWriter, r *http.Request) {
	rows, err := db.Query("SELECT * FROM counters")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer rows.Close()

	var n int
	var name string

	for rows.Next() {
		err = rows.Scan(&name, &n)
		if err != nil {
			fmt.Println(err)
		}
		fmt.Fprintf(w, "%s %d", name, n)
		fmt.Fprintln(w, "")
	}

	fmt.Fprint(w, "success")
}

func processRequest(data string) {
	//consume data
	time.Sleep(50 * time.Millisecond)
}

func incCounter(name string) {
	pageCountLock.Lock()
	pageCount[name]++
	pageCountLock.Unlock()
	storeCounterToDB()
}

func storeCounterToDB() {
	dbLock.Lock()
	defer dbLock.Unlock()
	for k, v := range pageCount {
		_, err := db.Exec("UPDATE counters SET n=? WHERE name=?", v, k)
		if err != nil {
			fmt.Println("failed to update counter", k, err)
		}
	}
}

func setupDB() {
	var err error
	db, err = sql.Open("sqlite3", "./counter.db")
	if err != nil {
		panic(err)
	}

	_, err = db.Exec("DROP TABLE IF EXISTS counters;")
	if err != nil {
		panic(err)
	}
	_, err = db.Exec("CREATE TABLE counters(name text, n integer);")
	if err != nil {
		panic(err)
	}

	for k, v := range pageCount {
		_, err := db.Exec("INSERT INTO counters(name, n) values (?,?)", k, v)
		if err != nil {
			panic(err)
		}
	}
}

func main() {
	setupDB()

	defer func() {
		if db != nil {
			db.Close()
		}
	}()

	http.HandleFunc("/", welcomeHandler)
	http.HandleFunc("/view/", viewHandler)
	http.HandleFunc("/counter/", counterHandler)
	http.HandleFunc("/counter-db/", counterDBHandler)

	http.ListenAndServe(":8080", nil)
}
