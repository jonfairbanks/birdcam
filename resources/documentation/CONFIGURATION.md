## Configuration

### BirdCam currently supports a few different features gated behind command line arguments.

<hr>

#### Debug Mode

`python3 main.py --debug`

Debug mode prints additional log items to console AND draws red bounding boxes around detected objects to assist in debugging.

<hr>

#### Slack Notifications

`python3 main.py --slack-token <SLACK BOT TOKEN>`

By passing a Slack bot token, BirdCam will upload snapshots of detected birds to Slack.

To prevent notification flood, there is currently a 60s delay between object detection events. This can be overridden by passing an overriding value: `--detection-delay=120`.

<hr>

#### Disabling Motion Detection

`python3 main.py --disable-motion`

Video stream only. Can be used if you do not want motion detection enabled.

<hr>

#### Override Web Port

`python3 main.py --port 8080`

The web stream runs on port 8000 by default but can be overridden if necessary.

<hr>
