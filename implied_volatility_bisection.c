//
//  implied_volatility_bisection.c
//  
//
//  Created by Domenico Natella on 10/19/16.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#define SIZE 109
#define MAX_ITERATIONS 1000000

struct option{
    double V_market[SIZE][2];
    double K[SIZE];
    double implied_vol[SIZE];
    double T;
    double S;
    double r;
};

struct tm create_tm(int year, int month, int day){
    struct tm my_time = { .tm_year=year, .tm_mon=month, .tm_mday=day,
        .tm_hour=0, .tm_min=0, .tm_sec=0 };
    return my_time;
}

struct option load(char* filename){
    FILE* file = fopen(filename, "r");
    struct option op;
    char tmp[12], cp[2];
    
    fscanf(file, "%lf", &op.S);
    
    fscanf(file, "%s", tmp);
    char s[2] = "/";
    char *token;
    token = strtok(tmp, s);
    int date[3]={0,0,0};
    int i = 0;
    while( token != NULL ){
        date[i] = atoi(token);
        token = strtok(NULL, s);
        i++;
    }
    time_t now;
    time(&now);
    struct tm option_t = create_tm(date[0]-1900, date[1]-1, date[2]);
    time_t opt_t_conv = mktime(&option_t);
    double diff_t = difftime(opt_t_conv, now);
    op.T = (diff_t/86400)/365.;
    
    i=0;
    while(fscanf(file, "%s", tmp)!=EOF){
        if(strcmp(tmp, "c")==0 | strcmp(tmp, "p")==0) strcpy(cp, tmp);
        else{
            op.K[i] = atof(strtok(tmp,s));
            op.V_market[i][0] = atof(strtok(NULL,s));
            if(strcmp(cp, "c")==0) op.V_market[i][1] = 0.;
            else if(strcmp(cp, "p")==0) op.V_market[i][1] = 1.;
        }
        i++;
    }
    
    op.r = 0.03;
    return op;
}

double cdf(double x){
    double RT2PI = sqrt(4.0*acos(0.0));
    static const double SPLIT = 7.07106781186547;
    static const double N0 = 220.206867912376;
    static const double N1 = 221.213596169931;
    static const double N2 = 112.079291497871;
    static const double N3 = 33.912866078383;
    static const double N4 = 6.37396220353165;
    static const double N5 = 0.700383064443688;
    static const double N6 = 3.52624965998911e-02;
    static const double M0 = 440.413735824752;
    static const double M1 = 793.826512519948;
    static const double M2 = 637.333633378831;
    static const double M3 = 296.564248779674;
    static const double M4 = 86.7807322029461;
    static const double M5 = 16.064177579207;
    static const double M6 = 1.75566716318264;
    static const double M7 = 8.83883476483184e-02;
    
    const double z = fabs(x);
    double c = 0.0;
    if(z<=37.0){
        const double e = exp(-z*z/2.0);
        if(z<SPLIT){
            const double n = (((((N6*z + N5)*z + N4)*z + N3)*z + N2)*z + N1)*z + N0;
            const double d = ((((((M7*z + M6)*z + M5)*z + M4)*z + M3)*z + M2)*z + M1)*z + M0;
            c = e*n/d;}
        else{
            const double f = z + 1.0/(z + 2.0/(z + 3.0/(z + 4.0/(z + 13.0/20.0))));
            c = e/(RT2PI*f);}
    }
    return x<=0.0 ? c : 1-c;
}

double d_j(int j, double S, double K, double r, double sigma, double T){
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*(pow(T,0.5)));
    if(j==1) return d1;
    else return d1-sigma*pow(T,0.5);
}

double call_price(double S, double K, double r, double sigma, double T, double type){
    if(type==0.) return S * cdf(d_j(1, S, K, r, sigma, T))-K*exp(-r*T) * cdf(d_j(2, S, K, r,sigma, T));
    else return K*exp(-r*T) * cdf(d_j(2, S, K, r,sigma, T)) - S * cdf(d_j(1, S, K, r, sigma, T));
}

double interval_bisection(double y_target, double m, double n, double epsilon, double S, double K, double r, double T, double type){
    int i=0;
    double x, y;
    do{
        if(i >= MAX_ITERATIONS) break;
        x = 0.5*(m+n);
        y = call_price(S, K, r, x, T,type);
        if(y<y_target) m = x;
        if(y>y_target) n = x;
        i++;
    }while (fabs(y-y_target) > epsilon);
    return x;
}


int main(int argc, char** argv){
    // First we create the parameter list
    // S: Underlying spot price
    // K: Strike price
    // r: Risk-free rate (5%)
    // T: One year until expiry
    // C_M: Option market price
    
    int rank,size, i;
    double low_vol = 0.3, high_vol = 3., epsilon = 0.001, t0, t1;
    struct option op[5], toReturn;
    
    int err = MPI_Init(&argc, &argv);
    t0 = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 1) {
        fprintf(stderr,"Requires at least two processes.\n");
        MPI_Finalize();
        exit(-1);
    }
    
    int blocklen[5] = {SIZE*2,SIZE,SIZE,2,2};
    MPI_Datatype types[6] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
    MPI_Datatype mpi_op_type;
    MPI_Aint offsets[5];
    
    offsets[0] = offsetof(struct option, V_market);
    offsets[1] = offsetof(struct option, K);
    offsets[2] = offsetof(struct option, implied_vol);
    offsets[3] = offsetof(struct option, T);
    offsets[4] = offsetof(struct option, S);
    offsets[5] = offsetof(struct option, r);
    
    MPI_Type_create_struct(6, blocklen, offsets, types, &mpi_op_type);
    MPI_Type_commit(&mpi_op_type);
    
    if(rank == 0){
        op[0] = load("./OPT_AAPL/Options_20161118.txt");
        op[1] = load("./OPT_AAPL/Options_2017120.txt");
        op[2] = load("./OPT_AAPL/Options_2017317.txt");
        op[3] = load("./OPT_AAPL/Options_2017421.txt");
        op[4] = load("./OPT_AAPL/Options_2017616.txt");
    }
    
    MPI_Scatter(&op,1,mpi_op_type,&toReturn,1,mpi_op_type,0,MPI_COMM_WORLD);
    
    printf("Processor %d has time: %.2f\n", rank, toReturn.T);
    
    for(i=0; i<10; i++){
        toReturn.implied_vol[i] = interval_bisection(toReturn.V_market[i][0], low_vol, high_vol, epsilon, toReturn.S, toReturn.K[i], toReturn.r, toReturn.T, toReturn.V_market[i][1]);
    }
    
    printf("Processor %d has implied vol.: %.2f\n", rank, toReturn.implied_vol[9]);
    
    MPI_Gather(&toReturn,1,mpi_op_type,&op,1,mpi_op_type,0,MPI_COMM_WORLD);
    
    if(rank == 0){
        int i, j;
        for(i=0; i<5; i++){
           printf("Implied vol. for time %.2f is %.2f%% \n",  (op[i].T), op[i].implied_vol[9]);
        }
        t1 = MPI_Wtime();
        printf("Time elapsed : %1.2f\n", t1-t0);
        fflush(stdout);
    }
    
    MPI_Type_free(&mpi_op_type);
    MPI_Finalize();
    return 0;
}


