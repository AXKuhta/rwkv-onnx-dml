#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <performance.h>

// Reports average ms/token and deviation
void report_performance(clock_t* timestamps, size_t count) {
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

	printf("%.1f ms/token, %.1f stddev\n", avg, stddev);
}
