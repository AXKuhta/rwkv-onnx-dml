#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <performance.h>

// Takes microseconds, reports average ms/token and deviation
// Please only feed it relative times
void report_performance(uint64_t* timestamps, size_t count) {
	double avg = 0.0;
	double avgsq = 0.0;
	double stddev = 0.0;

	for (size_t i = 0; i < count; i++) {
		avg += timestamps[i];
	}

	avg = avg / count;

	for (size_t i = 0; i < count; i++) {
		double centered = timestamps[i] - avg;
		avgsq += centered*centered;
	}

	avgsq = avgsq / count;
	stddev = sqrt(avgsq);

	printf("%.1f ms/token, %.1f ms stddev\n", avg / 1000.0, stddev / 1000.0);
}


uint64_t microseconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec*(uint64_t)1000000 + tv.tv_usec;
}

