## Hosted version

If you intend to focus on the client-side aspect of the problem set, you can start by leveraging the hosted version at <https://eq-work-samples-api.herokuapp.com>

## Environment

* Node.js 6.1+

## Setup and Run

0. Clone this repository `git clone git@github.com:EQWorks/work-samples.git` and change directory to `api-node`
1. Install Node level dependencies `$ npm install`
2. Set appropriate PostgreSQL environment variables according to https://www.postgresql.org/docs/9.6/static/libpq-envars.html
3. Run `$ npm run dev` and by default it should now be listening on port `5555`.
4. Open your browser and point to `localhost:5555` and you should see `Welcome to EQ Works 😎`

Alternatively, if you already have Docker setup and running:

0. Clone this repository and change directory to `api-node`
1. Store PostgreSQL environment variables into a `.env` file
2. Run `$ docker-compose up` (or `$ docker-compose up -d` to run in the background)
3. Open your browser and point to `localhost:5555` and you should see `Welcome to EQ Works 😎`

_Note_: you'll be given necessary PostgreSQL environment variable values along with the problem set.

## Notes on working through the problems

Make sure any external dependencies are properly added into `package.json`. Unless otherwise specified, we welcome a healthy mix of your own implementations, and good choices of existing open-source libraries/tools.

Your submission should be in the form of your local work sample repository packaged using [`git-archive`](https://git-scm.com/docs/git-archive) command. Do not include anything that's ignored by `.*ignore` files.
