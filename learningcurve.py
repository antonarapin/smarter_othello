import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import sys
import argparse

def smoothData(rawData, smooth):
    if smooth > 0:
        smoothed = [sum(rawData[0:smooth+1])/(smooth+1)]
        for i in range(smooth+1, len(rawData)):
            newSmoothed = smoothed[-1] - (rawData[i - smooth - 1] - rawData[i])/(smooth+1)
            smoothed.append(newSmoothed)

        return smoothed
    else:
        return rawData

parser = argparse.ArgumentParser(description='Display learning curves from the given file(s).')

parser.add_argument('files', metavar='file', nargs='+', help='file containing the score for each episode in a column.')
parser.add_argument('-c', '--column', type=int, default=1, help='the column in the files that contains the learning data (default: %(default)s).')
parser.add_argument('-s', '--smooth', type=int, default=1, help='the size of the smoothing window (default: %(default)s, which is no smoothing).')
parser.add_argument('-a', '--avg', action='store_true', default=False, help='display the average curve of the given files (rather than each curve individually). Episodes not contained in all files will be displayed in red.')
parser.add_argument('-r', '--raw', action='store_true', default=False, help='display the raw data as well as the smoothed data.')
parser.add_argument('-t', '--timesteps', metavar='COLUMN', type=int, default=0, help='uses the number of steps from the supplied column to display learning versus number of steps rather than number of episodes (has no effect when combined with -a).')
parser.add_argument('-l', '--title', metavar='TITLE', type=str, default='', help='The title to display for this graph.')

args = parser.parse_args()

column = args.column - 1
smooth = args.smooth - 1
stepCol = args.timesteps - 1

plot.ion();

data = []
steps = []
maxLength = 0
minLength = -1
for filename in args.files:
    data.append([])
    if stepCol >= 0:
        steps.append([])
        curStep = 0

    try:
        fin = open(filename, 'r')
        for line in fin:
            score = float(line.split()[column])
            if stepCol >= 0:
                step = int(line.split()[stepCol])
                curStep += step
                steps[-1].append(curStep)
            data[-1].append(score)
    except Exception as inst:
        sys.stderr.write("Error reading " + filename + "\n")
        sys.stderr.write(str(inst) + "\n")

    if len(data[-1]) > maxLength:
        maxLength = len(data[-1])

    if minLength < 0 or len(data[-1]) < minLength:
        minLength = len(data[-1])

if maxLength == 0:
    sys.stderr.write("No data to graph!\n")
    exit(1)

if args.avg:
    avgData = []
    for i in range(maxLength):
        total = 0
        count = 0
        for j in range(len(data)):
            if i < len(data[j]):
                total += data[j][i]
                count += 1
        avgData.append(total/count)

    if args.raw:
        plot.plot(range(minLength), avgData[:minLength], color='0.75', label='Avg. data')
        plot.plot(range(minLength, len(avgData)), avgData[minLength:], color='lightcoral')

    if minLength > smooth:
        smoothed = smoothData(avgData, smooth)
        plot.plot(range(smooth, minLength), smoothed[:minLength-smooth], color='black', label='Smoothed (window size ' + str(smooth + 1) + ')')
        plot.plot(range(minLength, len(avgData)), smoothed[minLength-smooth:], color='red')
else:
    if args.raw:
        for i in range(len(data)):
            rawData = data[i]
            xCoords = range(len(rawData))
            if stepCol >= 0:
                xCoords = steps[i]
            plot.plot(xCoords, rawData, color='0.75', label='Raw data')

    for i in range(len(data)):
        rawData = data[i]
        if len(rawData) > smooth:
            smoothed = smoothData(rawData, smooth)
            xCoords = range(smooth, len(rawData))
            if stepCol >= 0:
                xCoords = steps[i][smooth:]
            plot.plot(xCoords, smoothed, color='black', label='Smoothed (window size ' + str(smooth + 1) + ')')

plot.title("Learning")
if stepCol >= 0:
    plot.xlabel("Timestep")
else:
    plot.xlabel("Episode")
plot.ylabel("Score")
plot.title(args.title)

plot.draw()
raw_input("<press enter>")
