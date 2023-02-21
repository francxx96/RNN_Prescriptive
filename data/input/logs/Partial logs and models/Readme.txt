This folder contains the Declare models with which sub-logs were generated. 
Each sub-log was then merged in a single log (called Data-flow log), 
that is composed as follows:

Total number of traces: 2000

150 traces have the following structure:
	A B C D1 D2 E F(A) G1(X) H I

350 traces have the following structure:
	A B C D1 D2 E F(A) G3(X) H I

150 traces have the following structure:
	A B C D1 D2 E F(B) G2(X) H I

350 traces have the following structure:
	A B C D1 D2 E F(B) G3(X) H I

500 traces have the following structure:
	A B C D2 D1 E F(A) G2(Y) H I

500 traces have the following structure:
	A B C D2 D1 E F(B) G1(Y) H I