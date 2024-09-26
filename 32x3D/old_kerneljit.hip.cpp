JIT BEGIN
0 P10 594360 pointer_double
3 0 594360 2 
0 P11 594360 pointer_double
3 1 594360 2 
0 P1 594360 pointer_double
3 2 594360 2 
0 P3 491520 pointer_double
3 3 491520 2 
0 P7 594360 pointer_double
3 4 594360 2 
0 P8 594360 pointer_double
3 5 594360 2 
0 P5 594360 pointer_double
3 6 594360 2 
0 P9 594360 pointer_double
3 7 594360 2 
0 P2 274360 pointer_double
3 8 274360 2 
0 P6 594360 pointer_double
3 9 594360 2 
0 P4 594360 pointer_double
3 10 594360 2 
2 ker_code0 251 1 1 256 1 1 X -1 gamma1 -1 P1 -2
2 ker_code1 215 1 1 256 1 1 X -1 gamma1 -1 P4 -2
2 ker_code2 215 1 1 256 1 1 P1 -1 P2 -2 P4 -1
2 ker_code3 198 1 1 256 1 1 gamma1 -1 P1 -2 P2 -1
2 ker_code4 198 1 1 256 1 1 gamma1 -1 P1 -1 P3 -2
2 ker_code5 178 1 1 256 1 1 gamma1 -1 P1 -1 P5 -2
2 ker_code6 178 1 1 256 1 1 P1 -2 P3 -1 P5 -1
2 ker_code7 129 1 1 256 1 1 P1 -1 P3 -2
2 ker_code8 198 1 1 256 1 1 gamma1 -1 P8 -2 P2 -1
2 ker_code9 198 1 1 256 1 1 gamma1 -1 P10 -2 P8 -1
2 ker_code10 178 1 1 256 1 1 gamma1 -1 P8 -1 P6 -2
2 ker_code11 178 1 1 256 1 1 P10 -1 P8 -2 P6 -1
2 ker_code12 129 1 1 256 1 1 P10 -2 P8 -1
2 ker_code13 198 1 1 256 1 1 gamma1 -1 P9 -2 P2 -1
2 ker_code14 198 1 1 256 1 1 gamma1 -1 P11 -2 P9 -1
2 ker_code15 178 1 1 256 1 1 gamma1 -1 P7 -2 P9 -1
2 ker_code16 178 1 1 256 1 1 P11 -1 P9 -2 P7 -1
2 ker_code17 129 1 1 256 1 1 P11 -2 P9 -1
2 ker_code18 129 1 1 256 1 1 Y -2 a_scale1 -1 dx1 -1 P10 -1 P11 -1 P3 -1
4 Y pointer_double
4 X pointer_double
4 gamma1 double
4 a_scale1 double
4 dx1 double
------------------
#include "hip/hip_runtime.h"

extern "C" __global__ void ker_code0(double *X, double gamma1, double *P1) {
    if (((((256*blockIdx.x) + threadIdx.x) < 64000))) {
        double a427, a428, s28, s29, s30;
        int a422, a423, a424, a425, a426;
        a422 = (threadIdx.x + (256*blockIdx.x));
        s28 = X[a422];
        P1[a422] = s28;
        a423 = (a422 + 64000);
        s29 = (X[a423] / s28);
        P1[a423] = s29;
        a424 = (a422 + 128000);
        s30 = (X[a424] / s28);
        P1[a424] = s30;
        a425 = (a422 + 192000);
        P1[a425] = (X[a425] / s28);
        a426 = (a422 + 256000);
        a427 = (gamma1 - 1.0);
        a428 = (0.5*a427);
        P1[a426] = ((a427*X[a426]) - ((a428*((s29*s29)*s28)) + (a428*((s30*s30)*s28))));
    }
}

extern "C" __global__ void ker_code1(double *X, double gamma1, double *P4) {
    if (((((256*blockIdx.x) + threadIdx.x) < 54872))) {
        double a796, a797, s132, s133, s134, s135, s136, s137, 
                s138, t156;
        int a795, b101;
        a795 = ((256*blockIdx.x) + threadIdx.x);
        b101 = ((40*(a795 / 38)) + (((28*blockIdx.x) + threadIdx.x) % 38));
        s132 = X[(b101 + 1681)];
        t156 = (s132 - (0.041666666666666664*((X[(b101 + 41)] - (6.0*X[(b101 + 1641)])) + X[(b101 + 1601)] + X[(b101 + 1640)] + X[(b101 + 1642)] + s132 + X[(b101 + 3241)])));
        s133 = X[(b101 + 65681)];
        s134 = X[(b101 + 129681)];
        s135 = X[(b101 + 193681)];
        s136 = X[(b101 + 257681)];
        P4[(a795 + 320000)] = t156;
        s137 = ((s133 - (0.041666666666666664*((X[(b101 + 64041)] - (6.0*X[(b101 + 65641)])) + X[(b101 + 65601)] + X[(b101 + 65640)] + X[(b101 + 65642)] + s133 + X[(b101 + 67241)]))) / t156);
        P4[(a795 + 374872)] = s137;
        s138 = ((s134 - (0.041666666666666664*((X[(b101 + 128041)] - (6.0*X[(b101 + 129641)])) + X[(b101 + 129601)] + X[(b101 + 129640)] + X[(b101 + 129642)] + s134 + X[(b101 + 131241)]))) / t156);
        P4[(a795 + 429744)] = s138;
        P4[(a795 + 484616)] = ((s135 - (0.041666666666666664*((X[(b101 + 192041)] - (6.0*X[(b101 + 193641)])) + X[(b101 + 193601)] + X[(b101 + 193640)] + X[(b101 + 193642)] + s135 + X[(b101 + 195241)]))) / t156);
        a796 = (gamma1 - 1.0);
        a797 = (0.5*a796);
        P4[(a795 + 539488)] = ((a796*(s136 - (0.041666666666666664*((X[(b101 + 256041)] - (6.0*X[(b101 + 257641)])) + X[(b101 + 257601)] + X[(b101 + 257640)] + X[(b101 + 257642)] + s136 + X[(b101 + 259241)])))) - ((a797*((s137*s137)*t156)) + (a797*((s138*s138)*t156))));
    }
}

extern "C" __global__ void ker_code2(double *P1, double *P2, double *P4) {
    if (((((256*blockIdx.x) + threadIdx.x) < 54872))) {
        int a1165, b198;
        a1165 = ((256*blockIdx.x) + threadIdx.x);
        b198 = ((40*(a1165 / 38)) + (((28*blockIdx.x) + threadIdx.x) % 38));
        P2[a1165] = ((0.041666666666666664*((P1[(b198 + 41)] - (6.0*P1[(b198 + 1641)])) + P1[(b198 + 1601)] + P1[(b198 + 1640)] + P1[(b198 + 1642)] + P1[(b198 + 1681)] + P1[(b198 + 3241)])) + P4[(a1165 + 320000)]);
        P2[(a1165 + 54872)] = ((0.041666666666666664*((P1[(b198 + 64041)] - (6.0*P1[(b198 + 65641)])) + P1[(b198 + 65601)] + P1[(b198 + 65640)] + P1[(b198 + 65642)] + P1[(b198 + 65681)] + P1[(b198 + 67241)])) + P4[(a1165 + 374872)]);
        P2[(a1165 + 109744)] = ((0.041666666666666664*((P1[(b198 + 128041)] - (6.0*P1[(b198 + 129641)])) + P1[(b198 + 129601)] + P1[(b198 + 129640)] + P1[(b198 + 129642)] + P1[(b198 + 129681)] + P1[(b198 + 131241)])) + P4[(a1165 + 429744)]);
        P2[(a1165 + 164616)] = ((0.041666666666666664*((P1[(b198 + 192041)] - (6.0*P1[(b198 + 193641)])) + P1[(b198 + 193601)] + P1[(b198 + 193640)] + P1[(b198 + 193642)] + P1[(b198 + 193681)] + P1[(b198 + 195241)])) + P4[(a1165 + 484616)]);
        P2[(a1165 + 219488)] = ((0.041666666666666664*((P1[(b198 + 256041)] - (6.0*P1[(b198 + 257641)])) + P1[(b198 + 257601)] + P1[(b198 + 257640)] + P1[(b198 + 257642)] + P1[(b198 + 257681)] + P1[(b198 + 259241)])) + P4[(a1165 + 539488)]);
    }
}

extern "C" __global__ void ker_code3(double gamma1, double *P1, double *P2) {
    if (((((256*blockIdx.x) + threadIdx.x) < 50540))) {
        double s312, s313, s314, s315, s316, s317, s318, s319, 
                s320, s321, s322, s323, s324, s325, t213, t214, 
                t215, t216, t217, t218, t219, t220, t221, t222, 
                t223, t224, t225;
        int a1469, a1470, a1471, a1472, a1473, a1474, a1475, a1476, 
                a1477, a1478, a1479, a1480, a1481, a1482, a1483, a1484, 
                a1485, a1486, a1487, a1488, a1489, a1490, a1491, a1492, 
                a1493, a1494, a1495, a1496, a1497, a1498, a1499, a1500, 
                a1501, a1502;
        a1485 = ((256*blockIdx.x) + threadIdx.x);
        a1486 = ((38*(a1485 / 38)) + (((28*blockIdx.x) + threadIdx.x) % 38));
        t217 = (0.5*((((0.58333333333333337*P2[(a1486 + 54873)]) - (0.083333333333333329*P2[(a1486 + 54872)])) + (0.58333333333333337*P2[(a1486 + 54874)])) - (0.083333333333333329*P2[(a1486 + 54875)])));
        t218 = (0.5*((((0.58333333333333337*P2[(a1486 + 219489)]) - (0.083333333333333329*P2[(a1486 + 219488)])) + (0.58333333333333337*P2[(a1486 + 219490)])) - (0.083333333333333329*P2[(a1486 + 219491)])));
        t219 = (0.5*((((0.58333333333333337*P2[(a1486 + 1)]) - (0.083333333333333329*P2[a1486])) + (0.58333333333333337*P2[(a1486 + 2)])) - (0.083333333333333329*P2[(a1486 + 3)])));
        t220 = (t219 + t219);
        t221 = (t218 + t218);
        s318 = ((sqrt(gamma1)*sqrt(t221))*sqrt((1 / t220)));
        s319 = t221;
        if ((((t217 + t217) > 0))) {
            if ((((s318 - (t217 + t217)) > 0))) {
                t222 = ((s319 + (0.083333333333333329*P2[(a1486 + 219491)])) - (((0.58333333333333337*P2[(a1486 + 219489)]) - (0.083333333333333329*P2[(a1486 + 219488)])) + (0.58333333333333337*P2[(a1486 + 219490)])));
                s320 = (s318*s318);
                s321 = (1 / s320);
                s322 = (t222*s321);
                t223 = (s322 + ((((0.58333333333333337*P2[(a1486 + 1)]) - (0.083333333333333329*P2[a1486])) + (0.58333333333333337*P2[(a1486 + 2)])) - (0.083333333333333329*P2[(a1486 + 3)])));
                P1[a1485] = t223;
                a1487 = (a1485 + 50540);
                P1[a1487] = (t217 + t217);
                a1488 = (a1485 + 101080);
                P1[a1488] = ((((0.58333333333333337*P2[(a1486 + 109745)]) - (0.083333333333333329*P2[(a1486 + 109744)])) + (0.58333333333333337*P2[(a1486 + 109746)])) - (0.083333333333333329*P2[(a1486 + 109747)]));
                a1489 = (a1485 + 151620);
                P1[a1489] = ((((0.58333333333333337*P2[(a1486 + 164617)]) - (0.083333333333333329*P2[(a1486 + 164616)])) + (0.58333333333333337*P2[(a1486 + 164618)])) - (0.083333333333333329*P2[(a1486 + 164619)]));
                a1490 = (a1485 + 202160);
                P1[a1490] = s319;
            } else {
                P1[a1485] = ((((0.58333333333333337*P2[(a1486 + 1)]) - (0.083333333333333329*P2[a1486])) + (0.58333333333333337*P2[(a1486 + 2)])) - (0.083333333333333329*P2[(a1486 + 3)]));
                a1491 = (a1485 + 50540);
                P1[a1491] = ((((0.58333333333333337*P2[(a1486 + 54873)]) - (0.083333333333333329*P2[(a1486 + 54872)])) + (0.58333333333333337*P2[(a1486 + 54874)])) - (0.083333333333333329*P2[(a1486 + 54875)]));
                a1492 = (a1485 + 101080);
                P1[a1492] = ((((0.58333333333333337*P2[(a1486 + 109745)]) - (0.083333333333333329*P2[(a1486 + 109744)])) + (0.58333333333333337*P2[(a1486 + 109746)])) - (0.083333333333333329*P2[(a1486 + 109747)]));
                a1493 = (a1485 + 151620);
                P1[a1493] = ((((0.58333333333333337*P2[(a1486 + 164617)]) - (0.083333333333333329*P2[(a1486 + 164616)])) + (0.58333333333333337*P2[(a1486 + 164618)])) - (0.083333333333333329*P2[(a1486 + 164619)]));
                a1494 = (a1485 + 202160);
                P1[a1494] = ((((0.58333333333333337*P2[(a1486 + 219489)]) - (0.083333333333333329*P2[(a1486 + 219488)])) + (0.58333333333333337*P2[(a1486 + 219490)])) - (0.083333333333333329*P2[(a1486 + 219491)]));
            }
        } else {
            if ((((s318 + t217 + t217) > 0))) {
                t224 = ((s319 + (0.083333333333333329*P2[(a1486 + 219491)])) - (((0.58333333333333337*P2[(a1486 + 219489)]) - (0.083333333333333329*P2[(a1486 + 219488)])) + (0.58333333333333337*P2[(a1486 + 219490)])));
                s323 = (s318*s318);
                s324 = (1 / s323);
                s325 = (t224*s324);
                t225 = (s325 + ((((0.58333333333333337*P2[(a1486 + 1)]) - (0.083333333333333329*P2[a1486])) + (0.58333333333333337*P2[(a1486 + 2)])) - (0.083333333333333329*P2[(a1486 + 3)])));
                P1[a1485] = t225;
                a1495 = (a1485 + 50540);
                P1[a1495] = (t217 + t217);
                a1496 = (a1485 + 101080);
                P1[a1496] = ((((0.58333333333333337*P2[(a1486 + 109745)]) - (0.083333333333333329*P2[(a1486 + 109744)])) + (0.58333333333333337*P2[(a1486 + 109746)])) - (0.083333333333333329*P2[(a1486 + 109747)]));
                a1497 = (a1485 + 151620);
                P1[a1497] = ((((0.58333333333333337*P2[(a1486 + 164617)]) - (0.083333333333333329*P2[(a1486 + 164616)])) + (0.58333333333333337*P2[(a1486 + 164618)])) - (0.083333333333333329*P2[(a1486 + 164619)]));
                a1498 = (a1485 + 202160);
                P1[a1498] = s319;
            } else {
                P1[a1485] = ((((0.58333333333333337*P2[(a1486 + 1)]) - (0.083333333333333329*P2[a1486])) + (0.58333333333333337*P2[(a1486 + 2)])) - (0.083333333333333329*P2[(a1486 + 3)]));
                a1499 = (a1485 + 50540);
                P1[a1499] = ((((0.58333333333333337*P2[(a1486 + 54873)]) - (0.083333333333333329*P2[(a1486 + 54872)])) + (0.58333333333333337*P2[(a1486 + 54874)])) - (0.083333333333333329*P2[(a1486 + 54875)]));
                a1500 = (a1485 + 101080);
                P1[a1500] = ((((0.58333333333333337*P2[(a1486 + 109745)]) - (0.083333333333333329*P2[(a1486 + 109744)])) + (0.58333333333333337*P2[(a1486 + 109746)])) - (0.083333333333333329*P2[(a1486 + 109747)]));
                a1501 = (a1485 + 151620);
                P1[a1501] = ((((0.58333333333333337*P2[(a1486 + 164617)]) - (0.083333333333333329*P2[(a1486 + 164616)])) + (0.58333333333333337*P2[(a1486 + 164618)])) - (0.083333333333333329*P2[(a1486 + 164619)]));
                a1502 = (a1485 + 202160);
                P1[a1502] = ((((0.58333333333333337*P2[(a1486 + 219489)]) - (0.083333333333333329*P2[(a1486 + 219488)])) + (0.58333333333333337*P2[(a1486 + 219490)])) - (0.083333333333333329*P2[(a1486 + 219491)]));
            }
        }
    }
}

extern "C" __global__ void ker_code4(double gamma1, double *P1, double *P3) {
    if (((((256*blockIdx.x) + threadIdx.x) < 50540))) {
        double s361, s362, s363, s364, s365;
        int a1544, a1545, a1546, a1547, a1548;
        a1544 = ((256*blockIdx.x) + threadIdx.x);
        a1545 = (a1544 + 50540);
        s361 = P1[a1545];
        s362 = (P1[a1544]*s361);
        P3[a1544] = -(s362);
        a1546 = (a1544 + 202160);
        s363 = P1[a1546];
        P3[a1545] = -(((s362*s361) + s363));
        a1547 = (a1544 + 101080);
        s364 = P1[a1547];
        P3[a1547] = -((s362*s364));
        a1548 = (a1544 + 151620);
        s365 = P1[a1548];
        P3[a1548] = -((s362*s365));
        P3[a1546] = -((((gamma1 / (gamma1 - 1.0))*(s361*s363)) + (s362*(((0.5*s361)*s361) + ((0.5*s364)*s364) + ((0.5*s365)*s365)))));
    }
}

extern "C" __global__ void ker_code5(double gamma1, double *P1, double *P5) {
    if (((((256*blockIdx.x) + threadIdx.x) < 45360))) {
        double s458, t266, t267, t268, t269;
        int a1900, b331, b332;
        a1900 = ((256*blockIdx.x) + threadIdx.x);
        b331 = ((35*(a1900 / 35)) + (((11*blockIdx.x) + threadIdx.x) % 35));
        b332 = ((b331 % 1260) + (1330*(a1900 / 1260)));
        t266 = (P1[(b332 + 51905)] - (0.041666666666666664*((P1[(b331 + 50575)] - (4.0*P1[(b331 + 51905)])) + P1[(b331 + 51870)] + P1[(b331 + 51940)] + P1[(b331 + 53235)])));
        t267 = (P1[(b332 + 102445)] - (0.041666666666666664*((P1[(b331 + 101115)] - (4.0*P1[(b331 + 102445)])) + P1[(b331 + 102410)] + P1[(b331 + 102480)] + P1[(b331 + 103775)])));
        t268 = (P1[(b332 + 152985)] - (0.041666666666666664*((P1[(b331 + 151655)] - (4.0*P1[(b331 + 152985)])) + P1[(b331 + 152950)] + P1[(b331 + 153020)] + P1[(b331 + 154315)])));
        t269 = (P1[(b332 + 203525)] - (0.041666666666666664*((P1[(b331 + 202195)] - (4.0*P1[(b331 + 203525)])) + P1[(b331 + 203490)] + P1[(b331 + 203560)] + P1[(b331 + 204855)])));
        s458 = ((P1[(b332 + 1365)] - (0.041666666666666664*((P1[(b331 + 35)] - (4.0*P1[(b331 + 1365)])) + P1[(b331 + 1330)] + P1[(b331 + 1400)] + P1[(b331 + 2695)])))*t266);
        P5[(a1900 + 252700)] = -(s458);
        P5[(a1900 + 298060)] = -(((s458*t266) + t269));
        P5[(a1900 + 343420)] = -((s458*t267));
        P5[(a1900 + 388780)] = -((s458*t268));
        P5[(a1900 + 434140)] = -((((gamma1 / (gamma1 - 1.0))*(t266*t269)) + (s458*(((0.5*t266)*t266) + ((0.5*t267)*t267) + ((0.5*t268)*t268)))));
    }
}

extern "C" __global__ void ker_code6(double *P1, double *P3, double *P5) {
    if (((((256*blockIdx.x) + threadIdx.x) < 45360))) {
        int a2178, b400;
        a2178 = ((256*blockIdx.x) + threadIdx.x);
        b400 = ((35*(a2178 / 35)) + (((11*blockIdx.x) + threadIdx.x) % 35));
        P1[a2178] = ((0.041666666666666664*((P3[(b400 + 35)] - (4.0*P3[(b400 + 1365)])) + P3[(b400 + 1330)] + P3[(b400 + 1400)] + P3[(b400 + 2695)])) + P5[(a2178 + 252700)]);
        P1[(a2178 + 45360)] = ((0.041666666666666664*((P3[(b400 + 50575)] - (4.0*P3[(b400 + 51905)])) + P3[(b400 + 51870)] + P3[(b400 + 51940)] + P3[(b400 + 53235)])) + P5[(a2178 + 298060)]);
        P1[(a2178 + 90720)] = ((0.041666666666666664*((P3[(b400 + 101115)] - (4.0*P3[(b400 + 102445)])) + P3[(b400 + 102410)] + P3[(b400 + 102480)] + P3[(b400 + 103775)])) + P5[(a2178 + 343420)]);
        P1[(a2178 + 136080)] = ((0.041666666666666664*((P3[(b400 + 151655)] - (4.0*P3[(b400 + 152985)])) + P3[(b400 + 152950)] + P3[(b400 + 153020)] + P3[(b400 + 154315)])) + P5[(a2178 + 388780)]);
        P1[(a2178 + 181440)] = ((0.041666666666666664*((P3[(b400 + 202195)] - (4.0*P3[(b400 + 203525)])) + P3[(b400 + 203490)] + P3[(b400 + 203560)] + P3[(b400 + 204855)])) + P5[(a2178 + 434140)]);
    }
}

extern "C" __global__ void ker_code7(double *P1, double *P3) {
    if (((((256*blockIdx.x) + threadIdx.x) < 32768))) {
        int a2333, b417;
        a2333 = (threadIdx.x + (256*blockIdx.x));
        b417 = ((((35*(a2333 / 32)) + (threadIdx.x % 32)) % 1120) + (1260*(a2333 / 1024)));
        P3[a2333] = (P1[(b417 + 2592)] - P1[(b417 + 2591)]);
        P3[(a2333 + 32768)] = (P1[(b417 + 47952)] - P1[(b417 + 47951)]);
        P3[(a2333 + 65536)] = (P1[(b417 + 93312)] - P1[(b417 + 93311)]);
        P3[(a2333 + 98304)] = (P1[(b417 + 138672)] - P1[(b417 + 138671)]);
        P3[(a2333 + 131072)] = (P1[(b417 + 184032)] - P1[(b417 + 184031)]);
    }
}

extern "C" __global__ void ker_code8(double gamma1, double *P8, double *P2) {
    if (((((256*blockIdx.x) + threadIdx.x) < 50540))) {
        double s642, s643, s644, s645, s646, s647, s648, s649, 
                s650, s651, s652, s653, s654, s655, t326, t327, 
                t328, t329, t330, t331, t332, t333, t334, t335, 
                t336, t337, t338;
        int a2510, a2511, a2512, a2513, a2514, a2515, a2516, a2517, 
                a2518, a2519, a2520, a2521, a2522, a2523, a2524, a2525, 
                a2526, a2527, a2528, a2529, a2530, a2531, a2532, a2533, 
                a2534, a2535, a2536, a2537, a2538, a2539, a2540, a2541, 
                a2542;
        a2526 = ((256*blockIdx.x) + threadIdx.x);
        t330 = (0.5*((((0.58333333333333337*P2[(a2526 + 109782)]) - (0.083333333333333329*P2[(a2526 + 109744)])) + (0.58333333333333337*P2[(a2526 + 109820)])) - (0.083333333333333329*P2[(a2526 + 109858)])));
        t331 = (0.5*((((0.58333333333333337*P2[(a2526 + 219526)]) - (0.083333333333333329*P2[(a2526 + 219488)])) + (0.58333333333333337*P2[(a2526 + 219564)])) - (0.083333333333333329*P2[(a2526 + 219602)])));
        t332 = (0.5*((((0.58333333333333337*P2[(a2526 + 38)]) - (0.083333333333333329*P2[a2526])) + (0.58333333333333337*P2[(a2526 + 76)])) - (0.083333333333333329*P2[(a2526 + 114)])));
        t333 = (t332 + t332);
        t334 = (t331 + t331);
        s648 = ((sqrt(gamma1)*sqrt(t334))*sqrt((1 / t333)));
        s649 = t334;
        if ((((t330 + t330) > 0))) {
            if ((((s648 - (t330 + t330)) > 0))) {
                t335 = ((s649 + (0.083333333333333329*P2[(a2526 + 219602)])) - (((0.58333333333333337*P2[(a2526 + 219526)]) - (0.083333333333333329*P2[(a2526 + 219488)])) + (0.58333333333333337*P2[(a2526 + 219564)])));
                s650 = (s648*s648);
                s651 = (1 / s650);
                s652 = (t335*s651);
                t336 = (s652 + ((((0.58333333333333337*P2[(a2526 + 38)]) - (0.083333333333333329*P2[a2526])) + (0.58333333333333337*P2[(a2526 + 76)])) - (0.083333333333333329*P2[(a2526 + 114)])));
                P8[a2526] = t336;
                a2527 = (a2526 + 50540);
                P8[a2527] = ((((0.58333333333333337*P2[(a2526 + 54910)]) - (0.083333333333333329*P2[(a2526 + 54872)])) + (0.58333333333333337*P2[(a2526 + 54948)])) - (0.083333333333333329*P2[(a2526 + 54986)]));
                a2528 = (a2526 + 101080);
                P8[a2528] = (t330 + t330);
                a2529 = (a2526 + 151620);
                P8[a2529] = ((((0.58333333333333337*P2[(a2526 + 164654)]) - (0.083333333333333329*P2[(a2526 + 164616)])) + (0.58333333333333337*P2[(a2526 + 164692)])) - (0.083333333333333329*P2[(a2526 + 164730)]));
                a2530 = (a2526 + 202160);
                P8[a2530] = s649;
            } else {
                P8[a2526] = ((((0.58333333333333337*P2[(a2526 + 38)]) - (0.083333333333333329*P2[a2526])) + (0.58333333333333337*P2[(a2526 + 76)])) - (0.083333333333333329*P2[(a2526 + 114)]));
                a2531 = (a2526 + 50540);
                P8[a2531] = ((((0.58333333333333337*P2[(a2526 + 54910)]) - (0.083333333333333329*P2[(a2526 + 54872)])) + (0.58333333333333337*P2[(a2526 + 54948)])) - (0.083333333333333329*P2[(a2526 + 54986)]));
                a2532 = (a2526 + 101080);
                P8[a2532] = ((((0.58333333333333337*P2[(a2526 + 109782)]) - (0.083333333333333329*P2[(a2526 + 109744)])) + (0.58333333333333337*P2[(a2526 + 109820)])) - (0.083333333333333329*P2[(a2526 + 109858)]));
                a2533 = (a2526 + 151620);
                P8[a2533] = ((((0.58333333333333337*P2[(a2526 + 164654)]) - (0.083333333333333329*P2[(a2526 + 164616)])) + (0.58333333333333337*P2[(a2526 + 164692)])) - (0.083333333333333329*P2[(a2526 + 164730)]));
                a2534 = (a2526 + 202160);
                P8[a2534] = ((((0.58333333333333337*P2[(a2526 + 219526)]) - (0.083333333333333329*P2[(a2526 + 219488)])) + (0.58333333333333337*P2[(a2526 + 219564)])) - (0.083333333333333329*P2[(a2526 + 219602)]));
            }
        } else {
            if ((((s648 + t330 + t330) > 0))) {
                t337 = ((s649 + (0.083333333333333329*P2[(a2526 + 219602)])) - (((0.58333333333333337*P2[(a2526 + 219526)]) - (0.083333333333333329*P2[(a2526 + 219488)])) + (0.58333333333333337*P2[(a2526 + 219564)])));
                s653 = (s648*s648);
                s654 = (1 / s653);
                s655 = (t337*s654);
                t338 = (s655 + ((((0.58333333333333337*P2[(a2526 + 38)]) - (0.083333333333333329*P2[a2526])) + (0.58333333333333337*P2[(a2526 + 76)])) - (0.083333333333333329*P2[(a2526 + 114)])));
                P8[a2526] = t338;
                a2535 = (a2526 + 50540);
                P8[a2535] = ((((0.58333333333333337*P2[(a2526 + 54910)]) - (0.083333333333333329*P2[(a2526 + 54872)])) + (0.58333333333333337*P2[(a2526 + 54948)])) - (0.083333333333333329*P2[(a2526 + 54986)]));
                a2536 = (a2526 + 101080);
                P8[a2536] = (t330 + t330);
                a2537 = (a2526 + 151620);
                P8[a2537] = ((((0.58333333333333337*P2[(a2526 + 164654)]) - (0.083333333333333329*P2[(a2526 + 164616)])) + (0.58333333333333337*P2[(a2526 + 164692)])) - (0.083333333333333329*P2[(a2526 + 164730)]));
                a2538 = (a2526 + 202160);
                P8[a2538] = s649;
            } else {
                P8[a2526] = ((((0.58333333333333337*P2[(a2526 + 38)]) - (0.083333333333333329*P2[a2526])) + (0.58333333333333337*P2[(a2526 + 76)])) - (0.083333333333333329*P2[(a2526 + 114)]));
                a2539 = (a2526 + 50540);
                P8[a2539] = ((((0.58333333333333337*P2[(a2526 + 54910)]) - (0.083333333333333329*P2[(a2526 + 54872)])) + (0.58333333333333337*P2[(a2526 + 54948)])) - (0.083333333333333329*P2[(a2526 + 54986)]));
                a2540 = (a2526 + 101080);
                P8[a2540] = ((((0.58333333333333337*P2[(a2526 + 109782)]) - (0.083333333333333329*P2[(a2526 + 109744)])) + (0.58333333333333337*P2[(a2526 + 109820)])) - (0.083333333333333329*P2[(a2526 + 109858)]));
                a2541 = (a2526 + 151620);
                P8[a2541] = ((((0.58333333333333337*P2[(a2526 + 164654)]) - (0.083333333333333329*P2[(a2526 + 164616)])) + (0.58333333333333337*P2[(a2526 + 164692)])) - (0.083333333333333329*P2[(a2526 + 164730)]));
                a2542 = (a2526 + 202160);
                P8[a2542] = ((((0.58333333333333337*P2[(a2526 + 219526)]) - (0.083333333333333329*P2[(a2526 + 219488)])) + (0.58333333333333337*P2[(a2526 + 219564)])) - (0.083333333333333329*P2[(a2526 + 219602)]));
            }
        }
    }
}

extern "C" __global__ void ker_code9(double gamma1, double *P10, double *P8) {
    if (((((256*blockIdx.x) + threadIdx.x) < 50540))) {
        double s691, s692, s693, s694, s695;
        int a2584, a2585, a2586, a2587, a2588;
        a2584 = ((256*blockIdx.x) + threadIdx.x);
        a2585 = (a2584 + 101080);
        s691 = P8[a2585];
        s692 = (P8[a2584]*s691);
        P10[a2584] = -(s692);
        a2586 = (a2584 + 50540);
        s693 = P8[a2586];
        P10[a2586] = -((s692*s693));
        a2587 = (a2584 + 202160);
        s694 = P8[a2587];
        P10[a2585] = -(((s692*s691) + s694));
        a2588 = (a2584 + 151620);
        s695 = P8[a2588];
        P10[a2588] = -((s692*s695));
        P10[a2587] = -((((gamma1 / (gamma1 - 1.0))*(s691*s694)) + (s692*(((0.5*s693)*s693) + ((0.5*s691)*s691) + ((0.5*s695)*s695)))));
    }
}

extern "C" __global__ void ker_code10(double gamma1, double *P8, double *P6) {
    if (((((256*blockIdx.x) + threadIdx.x) < 45360))) {
        double s787, t379, t380, t381, t382;
        int a2940, b550, b551;
        a2940 = ((256*blockIdx.x) + threadIdx.x);
        b550 = ((38*(a2940 / 36)) + (((4*blockIdx.x) + threadIdx.x) % 36));
        b551 = ((b550 % 1330) + (1330*(a2940 / 1330)));
        t379 = (P8[(b551 + 51871)] - (0.041666666666666664*((P8[(b550 + 50541)] - (4.0*P8[(b550 + 51871)])) + P8[(b550 + 51870)] + P8[(b550 + 51872)] + P8[(b550 + 53201)])));
        t380 = (P8[(b551 + 102411)] - (0.041666666666666664*((P8[(b550 + 101081)] - (4.0*P8[(b550 + 102411)])) + P8[(b550 + 102410)] + P8[(b550 + 102412)] + P8[(b550 + 103741)])));
        t381 = (P8[(b551 + 152951)] - (0.041666666666666664*((P8[(b550 + 151621)] - (4.0*P8[(b550 + 152951)])) + P8[(b550 + 152950)] + P8[(b550 + 152952)] + P8[(b550 + 154281)])));
        t382 = (P8[(b551 + 203491)] - (0.041666666666666664*((P8[(b550 + 202161)] - (4.0*P8[(b550 + 203491)])) + P8[(b550 + 203490)] + P8[(b550 + 203492)] + P8[(b550 + 204821)])));
        s787 = ((P8[(b551 + 1331)] - (0.041666666666666664*((P8[(b550 + 1)] - (4.0*P8[(b550 + 1331)])) + P8[(b550 + 1330)] + P8[(b550 + 1332)] + P8[(b550 + 2661)])))*t380);
        P6[(a2940 + 252700)] = -(s787);
        P6[(a2940 + 298060)] = -((s787*t379));
        P6[(a2940 + 343420)] = -(((s787*t380) + t382));
        P6[(a2940 + 388780)] = -((s787*t381));
        P6[(a2940 + 434140)] = -((((gamma1 / (gamma1 - 1.0))*(t380*t382)) + (s787*(((0.5*t379)*t379) + ((0.5*t380)*t380) + ((0.5*t381)*t381)))));
    }
}

extern "C" __global__ void ker_code11(double *P10, double *P8, double *P6) {
    if (((((256*blockIdx.x) + threadIdx.x) < 45360))) {
        int a3218, b618;
        a3218 = ((256*blockIdx.x) + threadIdx.x);
        b618 = ((38*(a3218 / 36)) + (((4*blockIdx.x) + threadIdx.x) % 36));
        P8[a3218] = ((0.041666666666666664*((P10[(b618 + 1)] - (4.0*P10[(b618 + 1331)])) + P10[(b618 + 1330)] + P10[(b618 + 1332)] + P10[(b618 + 2661)])) + P6[(a3218 + 252700)]);
        P8[(a3218 + 45360)] = ((0.041666666666666664*((P10[(b618 + 50541)] - (4.0*P10[(b618 + 51871)])) + P10[(b618 + 51870)] + P10[(b618 + 51872)] + P10[(b618 + 53201)])) + P6[(a3218 + 298060)]);
        P8[(a3218 + 90720)] = ((0.041666666666666664*((P10[(b618 + 101081)] - (4.0*P10[(b618 + 102411)])) + P10[(b618 + 102410)] + P10[(b618 + 102412)] + P10[(b618 + 103741)])) + P6[(a3218 + 343420)]);
        P8[(a3218 + 136080)] = ((0.041666666666666664*((P10[(b618 + 151621)] - (4.0*P10[(b618 + 152951)])) + P10[(b618 + 152950)] + P10[(b618 + 152952)] + P10[(b618 + 154281)])) + P6[(a3218 + 388780)]);
        P8[(a3218 + 181440)] = ((0.041666666666666664*((P10[(b618 + 202161)] - (4.0*P10[(b618 + 203491)])) + P10[(b618 + 203490)] + P10[(b618 + 203492)] + P10[(b618 + 204821)])) + P6[(a3218 + 434140)]);
    }
}

extern "C" __global__ void ker_code12(double *P10, double *P8) {
    if (((((256*blockIdx.x) + threadIdx.x) < 32768))) {
        int a3374, b636;
        a3374 = (threadIdx.x + (256*blockIdx.x));
        b636 = ((((36*(a3374 / 32)) + (threadIdx.x % 32)) % 1152) + (1260*(a3374 / 1024)));
        P10[(a3374 + 163840)] = (P8[(b636 + 2628)] - P8[(b636 + 2592)]);
        P10[(a3374 + 196608)] = (P8[(b636 + 47988)] - P8[(b636 + 47952)]);
        P10[(a3374 + 229376)] = (P8[(b636 + 93348)] - P8[(b636 + 93312)]);
        P10[(a3374 + 262144)] = (P8[(b636 + 138708)] - P8[(b636 + 138672)]);
        P10[(a3374 + 294912)] = (P8[(b636 + 184068)] - P8[(b636 + 184032)]);
    }
}

extern "C" __global__ void ker_code13(double gamma1, double *P9, double *P2) {
    if (((((256*blockIdx.x) + threadIdx.x) < 50540))) {
        double s971, s972, s973, s974, s975, s976, s977, s978, 
                s979, s980, s981, s982, s983, s984, t440, t441, 
                t442, t443, t444, t445, t446, t447, t448, t449, 
                t450, t451, t452;
        int a3551, a3552, a3553, a3554, a3555, a3556, a3557, a3558, 
                a3559, a3560, a3561, a3562, a3563, a3564, a3565, a3566, 
                a3567, a3568, a3569, a3570, a3571, a3572, a3573, a3574, 
                a3575, a3576, a3577, a3578, a3579, a3580, a3581, a3582, 
                a3583;
        a3567 = ((256*blockIdx.x) + threadIdx.x);
        t444 = (0.5*((((0.58333333333333337*P2[(a3567 + 166060)]) - (0.083333333333333329*P2[(a3567 + 164616)])) + (0.58333333333333337*P2[(a3567 + 167504)])) - (0.083333333333333329*P2[(a3567 + 168948)])));
        t445 = (0.5*((((0.58333333333333337*P2[(a3567 + 220932)]) - (0.083333333333333329*P2[(a3567 + 219488)])) + (0.58333333333333337*P2[(a3567 + 222376)])) - (0.083333333333333329*P2[(a3567 + 223820)])));
        t446 = (0.5*((((0.58333333333333337*P2[(a3567 + 1444)]) - (0.083333333333333329*P2[a3567])) + (0.58333333333333337*P2[(a3567 + 2888)])) - (0.083333333333333329*P2[(a3567 + 4332)])));
        t447 = (t446 + t446);
        t448 = (t445 + t445);
        s977 = ((sqrt(gamma1)*sqrt(t448))*sqrt((1 / t447)));
        s978 = t448;
        if ((((t444 + t444) > 0))) {
            if ((((s977 - (t444 + t444)) > 0))) {
                t449 = ((s978 + (0.083333333333333329*P2[(a3567 + 223820)])) - (((0.58333333333333337*P2[(a3567 + 220932)]) - (0.083333333333333329*P2[(a3567 + 219488)])) + (0.58333333333333337*P2[(a3567 + 222376)])));
                s979 = (s977*s977);
                s980 = (1 / s979);
                s981 = (t449*s980);
                t450 = (s981 + ((((0.58333333333333337*P2[(a3567 + 1444)]) - (0.083333333333333329*P2[a3567])) + (0.58333333333333337*P2[(a3567 + 2888)])) - (0.083333333333333329*P2[(a3567 + 4332)])));
                P9[a3567] = t450;
                a3568 = (a3567 + 50540);
                P9[a3568] = ((((0.58333333333333337*P2[(a3567 + 56316)]) - (0.083333333333333329*P2[(a3567 + 54872)])) + (0.58333333333333337*P2[(a3567 + 57760)])) - (0.083333333333333329*P2[(a3567 + 59204)]));
                a3569 = (a3567 + 101080);
                P9[a3569] = ((((0.58333333333333337*P2[(a3567 + 111188)]) - (0.083333333333333329*P2[(a3567 + 109744)])) + (0.58333333333333337*P2[(a3567 + 112632)])) - (0.083333333333333329*P2[(a3567 + 114076)]));
                a3570 = (a3567 + 151620);
                P9[a3570] = (t444 + t444);
                a3571 = (a3567 + 202160);
                P9[a3571] = s978;
            } else {
                P9[a3567] = ((((0.58333333333333337*P2[(a3567 + 1444)]) - (0.083333333333333329*P2[a3567])) + (0.58333333333333337*P2[(a3567 + 2888)])) - (0.083333333333333329*P2[(a3567 + 4332)]));
                a3572 = (a3567 + 50540);
                P9[a3572] = ((((0.58333333333333337*P2[(a3567 + 56316)]) - (0.083333333333333329*P2[(a3567 + 54872)])) + (0.58333333333333337*P2[(a3567 + 57760)])) - (0.083333333333333329*P2[(a3567 + 59204)]));
                a3573 = (a3567 + 101080);
                P9[a3573] = ((((0.58333333333333337*P2[(a3567 + 111188)]) - (0.083333333333333329*P2[(a3567 + 109744)])) + (0.58333333333333337*P2[(a3567 + 112632)])) - (0.083333333333333329*P2[(a3567 + 114076)]));
                a3574 = (a3567 + 151620);
                P9[a3574] = ((((0.58333333333333337*P2[(a3567 + 166060)]) - (0.083333333333333329*P2[(a3567 + 164616)])) + (0.58333333333333337*P2[(a3567 + 167504)])) - (0.083333333333333329*P2[(a3567 + 168948)]));
                a3575 = (a3567 + 202160);
                P9[a3575] = ((((0.58333333333333337*P2[(a3567 + 220932)]) - (0.083333333333333329*P2[(a3567 + 219488)])) + (0.58333333333333337*P2[(a3567 + 222376)])) - (0.083333333333333329*P2[(a3567 + 223820)]));
            }
        } else {
            if ((((s977 + t444 + t444) > 0))) {
                t451 = ((s978 + (0.083333333333333329*P2[(a3567 + 223820)])) - (((0.58333333333333337*P2[(a3567 + 220932)]) - (0.083333333333333329*P2[(a3567 + 219488)])) + (0.58333333333333337*P2[(a3567 + 222376)])));
                s982 = (s977*s977);
                s983 = (1 / s982);
                s984 = (t451*s983);
                t452 = (s984 + ((((0.58333333333333337*P2[(a3567 + 1444)]) - (0.083333333333333329*P2[a3567])) + (0.58333333333333337*P2[(a3567 + 2888)])) - (0.083333333333333329*P2[(a3567 + 4332)])));
                P9[a3567] = t452;
                a3576 = (a3567 + 50540);
                P9[a3576] = ((((0.58333333333333337*P2[(a3567 + 56316)]) - (0.083333333333333329*P2[(a3567 + 54872)])) + (0.58333333333333337*P2[(a3567 + 57760)])) - (0.083333333333333329*P2[(a3567 + 59204)]));
                a3577 = (a3567 + 101080);
                P9[a3577] = ((((0.58333333333333337*P2[(a3567 + 111188)]) - (0.083333333333333329*P2[(a3567 + 109744)])) + (0.58333333333333337*P2[(a3567 + 112632)])) - (0.083333333333333329*P2[(a3567 + 114076)]));
                a3578 = (a3567 + 151620);
                P9[a3578] = (t444 + t444);
                a3579 = (a3567 + 202160);
                P9[a3579] = s978;
            } else {
                P9[a3567] = ((((0.58333333333333337*P2[(a3567 + 1444)]) - (0.083333333333333329*P2[a3567])) + (0.58333333333333337*P2[(a3567 + 2888)])) - (0.083333333333333329*P2[(a3567 + 4332)]));
                a3580 = (a3567 + 50540);
                P9[a3580] = ((((0.58333333333333337*P2[(a3567 + 56316)]) - (0.083333333333333329*P2[(a3567 + 54872)])) + (0.58333333333333337*P2[(a3567 + 57760)])) - (0.083333333333333329*P2[(a3567 + 59204)]));
                a3581 = (a3567 + 101080);
                P9[a3581] = ((((0.58333333333333337*P2[(a3567 + 111188)]) - (0.083333333333333329*P2[(a3567 + 109744)])) + (0.58333333333333337*P2[(a3567 + 112632)])) - (0.083333333333333329*P2[(a3567 + 114076)]));
                a3582 = (a3567 + 151620);
                P9[a3582] = ((((0.58333333333333337*P2[(a3567 + 166060)]) - (0.083333333333333329*P2[(a3567 + 164616)])) + (0.58333333333333337*P2[(a3567 + 167504)])) - (0.083333333333333329*P2[(a3567 + 168948)]));
                a3583 = (a3567 + 202160);
                P9[a3583] = ((((0.58333333333333337*P2[(a3567 + 220932)]) - (0.083333333333333329*P2[(a3567 + 219488)])) + (0.58333333333333337*P2[(a3567 + 222376)])) - (0.083333333333333329*P2[(a3567 + 223820)]));
            }
        }
    }
}

extern "C" __global__ void ker_code14(double gamma1, double *P11, double *P9) {
    if (((((256*blockIdx.x) + threadIdx.x) < 50540))) {
        double s1020, s1021, s1022, s1023, s1024;
        int a3625, a3626, a3627, a3628, a3629;
        a3625 = ((256*blockIdx.x) + threadIdx.x);
        a3626 = (a3625 + 151620);
        s1020 = P9[a3626];
        s1021 = (P9[a3625]*s1020);
        P11[a3625] = -(s1021);
        a3627 = (a3625 + 50540);
        s1022 = P9[a3627];
        P11[a3627] = -((s1021*s1022));
        a3628 = (a3625 + 101080);
        s1023 = P9[a3628];
        P11[a3628] = -((s1021*s1023));
        a3629 = (a3625 + 202160);
        s1024 = P9[a3629];
        P11[a3626] = -(((s1021*s1020) + s1024));
        P11[a3629] = -((((gamma1 / (gamma1 - 1.0))*(s1020*s1024)) + (s1021*(((0.5*s1022)*s1022) + ((0.5*s1023)*s1023) + ((0.5*s1020)*s1020)))));
    }
}

extern "C" __global__ void ker_code15(double gamma1, double *P7, double *P9) {
    if (((((256*blockIdx.x) + threadIdx.x) < 45360))) {
        double s1116, t493, t494, t495, t496;
        int a3981, b769, b770;
        a3981 = ((256*blockIdx.x) + threadIdx.x);
        b769 = ((38*(a3981 / 36)) + (((4*blockIdx.x) + threadIdx.x) % 36));
        b770 = ((b769 % 1368) + (1444*(a3981 / 1296)));
        t493 = (P9[(b770 + 50579)] - (0.041666666666666664*((P9[(b769 + 50541)] - (4.0*P9[(b769 + 50579)])) + P9[(b769 + 50578)] + P9[(b769 + 50580)] + P9[(b769 + 50617)])));
        t494 = (P9[(b770 + 101119)] - (0.041666666666666664*((P9[(b769 + 101081)] - (4.0*P9[(b769 + 101119)])) + P9[(b769 + 101118)] + P9[(b769 + 101120)] + P9[(b769 + 101157)])));
        t495 = (P9[(b770 + 151659)] - (0.041666666666666664*((P9[(b769 + 151621)] - (4.0*P9[(b769 + 151659)])) + P9[(b769 + 151658)] + P9[(b769 + 151660)] + P9[(b769 + 151697)])));
        t496 = (P9[(b770 + 202199)] - (0.041666666666666664*((P9[(b769 + 202161)] - (4.0*P9[(b769 + 202199)])) + P9[(b769 + 202198)] + P9[(b769 + 202200)] + P9[(b769 + 202237)])));
        s1116 = ((P9[(b770 + 39)] - (0.041666666666666664*((P9[(b769 + 1)] - (4.0*P9[(b769 + 39)])) + P9[(b769 + 38)] + P9[(b769 + 40)] + P9[(b769 + 77)])))*t495);
        P7[(a3981 + 252700)] = -(s1116);
        P7[(a3981 + 298060)] = -((s1116*t493));
        P7[(a3981 + 343420)] = -((s1116*t494));
        P7[(a3981 + 388780)] = -(((s1116*t495) + t496));
        P7[(a3981 + 434140)] = -((((gamma1 / (gamma1 - 1.0))*(t495*t496)) + (s1116*(((0.5*t493)*t493) + ((0.5*t494)*t494) + ((0.5*t495)*t495)))));
    }
}

extern "C" __global__ void ker_code16(double *P11, double *P9, double *P7) {
    if (((((256*blockIdx.x) + threadIdx.x) < 45360))) {
        int a4259, b837;
        a4259 = ((256*blockIdx.x) + threadIdx.x);
        b837 = ((38*(a4259 / 36)) + (((4*blockIdx.x) + threadIdx.x) % 36));
        P9[a4259] = ((0.041666666666666664*((P11[(b837 + 1)] - (4.0*P11[(b837 + 39)])) + P11[(b837 + 38)] + P11[(b837 + 40)] + P11[(b837 + 77)])) + P7[(a4259 + 252700)]);
        P9[(a4259 + 45360)] = ((0.041666666666666664*((P11[(b837 + 50541)] - (4.0*P11[(b837 + 50579)])) + P11[(b837 + 50578)] + P11[(b837 + 50580)] + P11[(b837 + 50617)])) + P7[(a4259 + 298060)]);
        P9[(a4259 + 90720)] = ((0.041666666666666664*((P11[(b837 + 101081)] - (4.0*P11[(b837 + 101119)])) + P11[(b837 + 101118)] + P11[(b837 + 101120)] + P11[(b837 + 101157)])) + P7[(a4259 + 343420)]);
        P9[(a4259 + 136080)] = ((0.041666666666666664*((P11[(b837 + 151621)] - (4.0*P11[(b837 + 151659)])) + P11[(b837 + 151658)] + P11[(b837 + 151660)] + P11[(b837 + 151697)])) + P7[(a4259 + 388780)]);
        P9[(a4259 + 181440)] = ((0.041666666666666664*((P11[(b837 + 202161)] - (4.0*P11[(b837 + 202199)])) + P11[(b837 + 202198)] + P11[(b837 + 202200)] + P11[(b837 + 202237)])) + P7[(a4259 + 434140)]);
    }
}

extern "C" __global__ void ker_code17(double *P11, double *P9) {
    if (((((256*blockIdx.x) + threadIdx.x) < 32768))) {
        int a4415, b855;
        a4415 = (threadIdx.x + (256*blockIdx.x));
        b855 = ((((36*(a4415 / 32)) + (threadIdx.x % 32)) % 1152) + (1296*(a4415 / 1024)));
        P11[(a4415 + 327680)] = (P9[(b855 + 3960)] - P9[(b855 + 2664)]);
        P11[(a4415 + 360448)] = (P9[(b855 + 49320)] - P9[(b855 + 48024)]);
        P11[(a4415 + 393216)] = (P9[(b855 + 94680)] - P9[(b855 + 93384)]);
        P11[(a4415 + 425984)] = (P9[(b855 + 140040)] - P9[(b855 + 138744)]);
        P11[(a4415 + 458752)] = (P9[(b855 + 185400)] - P9[(b855 + 184104)]);
    }
}

extern "C" __global__ void ker_code18(double *Y, double a_scale1, double dx1, double *P10, double *P11, double *P3) {
    if (((((256*blockIdx.x) + threadIdx.x) < 32768))) {
        double a4485;
        int a4484, a4486, a4487, a4488, a4489;
        a4484 = (threadIdx.x + (256*blockIdx.x));
        a4485 = (a_scale1 / dx1);
        Y[a4484] = (a4485*(P3[a4484] + P10[(a4484 + 163840)] + P11[(a4484 + 163840)]));
        a4486 = (a4484 + 32768);
        Y[a4486] = (a4485*(P3[a4486] + P10[(a4484 + 196608)] + P11[(a4484 + 196608)]));
        a4487 = (a4484 + 65536);
        Y[a4487] = (a4485*(P3[a4487] + P10[(a4484 + 229376)] + P11[(a4484 + 229376)]));
        a4488 = (a4484 + 98304);
        Y[a4488] = (a4485*(P3[a4488] + P10[(a4484 + 262144)] + P11[(a4484 + 262144)]));
        a4489 = (a4484 + 131072);
        Y[a4489] = (a4485*(P3[a4489] + P10[(a4484 + 294912)] + P11[(a4484 + 294912)]));
    }
}
