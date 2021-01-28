To build a docker image and run a demo:

1. Install [docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).
2. Run "`docker-compose up`" (if executed from `adl_2020/demo`) or "`docker-compose -f demo/docker-compose.yaml up`" (if executed from `adl_2020`) to build (if not been done yet) and run the image.

   To rebuild and run the image again - add "`--build`" to the end of the previous command.
4. Open `http://localhost:8866/` in a browser.