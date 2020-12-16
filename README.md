# openvino-callback-issue
Short illustration of single-threaded callback issue using Python asynchronous requests with OpenVINO

## Building
docker build -t openvino-callback-issue:latest .

## Test Running

### Working Test
docker run --rm -it openvino-callback-issue:latest 1

This works because there is only one request blocking the call-back thread at a given time.  The main producer thread feeding it work just waits until it gets through an item then gives it more work to do.

### Failing Test
docker run --rm -it openvino-callback-issue:latest 2

This fails because there are multiple async requests and the one that is done first will block the one call-back thread causing the second call-back to never occur.  This results in the main producer thread deadlocked waiting for both work items it provided to be completed.