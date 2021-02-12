#include <iostream>
#include <cmath>

using namespace std;

double frand() // ������������ ��������� ����� � ��������� [0,1)
{
	return double(rand())/RAND_MAX;
}

int eval(int* a, int n)
{
	int sum = 0;
	for( int i=0; i<n; i++ )
		sum += a[i];
	return sum;
}

int sphere(int * a, int n)
{
    int sum = 0;
    for(int i = 0; i < n ;i++){
        sum += a[i] * a[i];
    }
    return sum;
}

void init(int* P, int m, int n)
{
	for( int k=0; k<m; k++ ){
		for( int i=0; i<n; i++ ) {
			P[k*n+i] = rand()%2;
			cout << P[k * n + i] << " ";
		}
		cout << endl;
	}
}

void shuffle(int* P, int m, int n)
{
	for( int k=0; k<m; k++ )
	{
		int l = rand()%m;
		for( int i=0; i<n; i++ )
			swap(P[k*n+i],P[l*n+i]);
	}
}

void select(int* P, int m, int n)
{
	double pwin = 0.75;
	shuffle(P, m, n);
	for( int k=0; k<m/2; k++ )
	{
		int a = 2*k;
		int b = 2*k+1;
		int fa = sphere(P+a*n, n);
		int fb = sphere(P+b*n, n);
		double p = rand();
		if( fa<fb && p<pwin || fa => fb && p => pwin )
			for( int i=0; i<n; i++ )
				P[b*n+i] = P[a*n+i];
		else
			for( int i=0; i<n; i++ )
				P[a*n+i] = P[b*n+i];
	}
}

void crossover(int* P, int m, int n)
{
	shuffle(P, m, n);
	for( int k=0; k<m/2; k++ )
	{
		int a = 2*k;
		int b = 2*k+1;
		int j = rand()%n;
		for( int i=j; i<n; i++ )
			swap(P[a*n+i],P[b*n+i]);
	}
}

void mutate(int* P, int m, int n)
{
	double pmut = 0.1;
	for( int k=0; k<m; k++ )
		for( int i=0; i<n; i++ )
			if( rand()<pmut )
				P[k*n+i] = 1-P[k*n+i];
}

void printthebest(int* P, int m, int n)
{
	int k0 = -1;
	int f0 = -1;
	for( int k=0; k<m; k++)
	{
		int f = sphere(P+k*n, n);
		if( f>f0 )
		{
			f0 = f;
			k0 = k;
		}
	}
	cout << f0 << ": ";
	for( int i=0; i<n; i++ )
		cout << P[k0*n+i];
	cout << endl;
}

void runGA(int n, int m, int T)
{
	int* P = new int[n*m];
	init(P, m, n);
	for( int t=0; t<T; t++ )
	{
		select(P, m, n);
		crossover(P, m, n);
		mutate(P, m, n);
		//printthebest(P, m, n);
	}
	printthebest(P, m, n);

	delete[] P;
}

int main(int argc, char** argv)
{
	//int n = 10; //atoi(argv[1]);
	//int m = 20; // atoi(argv[2]);
	//int T = 10; // atoi(argv[3]);
	int n = 20; //atoi(argv[1]);
	int m = 6000; // atoi(argv[2]);
	int T = 2000; // atoi(argv[3]);

	runGA(n, m, T);
	getchar();
	return 0;
}