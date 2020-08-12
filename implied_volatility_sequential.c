//
//  implied_volatility_sequential.c
//  
//
//  Created by Domenico Natella on 11/1/16.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define SIZE 15
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
    
    fscanf(file, "%lf", &op.S);
    
    char tmp[12], cp[2];
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
            else op.V_market[i][1] = 0.;
        }
        i++;
    }
    op.r = 0.03;
    return op;
}

double pdf(const double x) {
    return (1.0/(pow(2*M_PI,0.5)))*exp(-0.5*x*x);
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
    else return K*exp(-r*T) * cdf(d_j(2, S, K, r,sigma, T)) - S * cdf(d_j(1, S, K, r, sigma, T)) ;
}

double call_vega(const double S, const double K, const double r, const double sigma, const double T) {
    return S * sqrt(T) * pdf(d_j(1, S, K, r, sigma, T));
}

double newton_raphson(double y_target, double init, double epsilon, double S, double K, double r, double T, double type){
    double x = init;
    double y = call_price(S, K, r, x, T,type);
    int i=0;
    
    while (fabs(y-y_target) > epsilon) {
        if(i >= MAX_ITERATIONS) break;
        double d_x = call_vega(S, K, r, x, T);
        x += (y-y_target)/d_x;
        y = call_price(S,K,r,x,T,type);
        i++;
    }
    return x;
}

int main(int argc, char** argv){
    struct timeval start, end;
    gettimeofday(&start, NULL);

    int i,j, num_iteration=14;
    double low_vol = 0.3, epsilon = 0.001, t0, t1;
    struct option op[7];
    
    op[0] = load("./OPT_AAPL/Options_20161118.txt");
    op[1] = load("./OPT_AAPL/Options_2017120.txt");
    op[2] = load("./OPT_AAPL/Options_2017317.txt");
    op[3] = load("./OPT_AAPL/Options_2017421.txt");
    op[4] = load("./OPT_AAPL/Options_2017616.txt");
    op[5] = load("./OPT_AAPL/Options_20171117.txt");
    op[6] = load("./OPT_AAPL/Options_2018119.txt");
    
    
    for(i=0; i<(sizeof(op)/sizeof(op[0])); i++){
        for(j=0; j<num_iteration; j++){
            printf("Implied vol. for time: %.2f: %.2f\n", op[i].T, newton_raphson(op[i].V_market[j][0], low_vol, epsilon, op[i].S, op[i].K[j], op[i].r, op[i].T, op[i].V_market[j][1]));
        }
    }
    
    gettimeofday(&end, NULL);
    printf("Wall time elapsed: %ld", (end.tv_sec  - start.tv_sec));
}
