## Environment

* Golang 1.8+

## Setup and Run

To setup environment:

- Install latest golang version
- Set the $GOPATH to point to project folder (../work-samples/backend-go/)
- Follow installation instructions from https://github.com/mattn/go-sqlite3
- Install Apache ab tool (for Ubuntu `sudo apt-get install apache2-utils`)

To run the app:

- Build the web server `go install counter`
- Start the server `../bin/counter`
- With the app running use `ab` tool to run benchmarks `ab -k -n 500 -c 2 http://localhost:8080/view/`
- Check the pages `http://localhost:8080/counter/` and `http://localhost:8080/counter-db/`

Alternatively, if you already have Docker setup and running:

- Run `$ docker-compose up` (or `$ docker-compose up -d` to run in the background)
- Check the pages `http://localhost:8080/counter/` and `http://localhost:8080/counter-db/`
