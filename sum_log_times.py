total_time = 0.0

with open('log.txt', 'r') as ctx:
    # Remove the \n from the end of each line
    lines = [a[:-1] for a in ctx.readlines()]

for line in lines:
    time = line.split(' ')[-1]
    if ':' in time:
        # time is in the format minutes:seconds
        spl = time.split(':')
        duration = 60 * int(spl[0]) + int(spl[1])
    else:
        # time is in the format ##.###s
        duration = float(time[:-1])
    total_time += duration
print('Total runtime: {:.2f}s'.format(total_time))