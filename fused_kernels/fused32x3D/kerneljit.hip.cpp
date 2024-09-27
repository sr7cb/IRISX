JIT BEGIN
0 P2 758100 pointer_double
3 0 758100 2 
0 P1 680400 pointer_double
3 1 680400 2 
2 ker_code0 215 1 1 256 1 1 X -1 gamma1 -1 P1 -2
2 ker_code1 198 1 1 256 1 1 gamma1 -1 P2 -2 P1 -1
2 ker_code2 178 1 1 256 1 1 gamma1 -1 P2 -1 P1 -2
2 ker_code3 129 1 1 256 1 1 P2 -2 P1 -1
2 ker_code4 129 1 1 256 1 1 Y -2 a_scale1 -1 dx1 -1 P2 -1
4 Y pointer_double
4 X pointer_double
4 gamma1 double
4 a_scale1 double
4 dx1 double
------------------
#include "hip/hip_runtime.h"

extern "C" __global__ void ker_code0(double *X, double gamma1, double *P1) {
    if (((((256*blockIdx.x) + threadIdx.x) < 54872))) {
        double a1245, a1246, s292, s293, s294, s295, s296, s297, 
                s298, s299, s300, s301, s302, s303, s304, s305, 
                s306, s307, s308, s309, s310, s311, s312, s313, 
                s314, s315, s316, s317, s318, s319, s320, s321, 
                s322, s323, s324, s325, s326, s327, s328, s329, 
                s330, s331, s332, s333, s334, s335, s336, s337, 
                s338, s339, s340, s341, s342, s343, s344, s345, 
                s346, s347, s348, s349, s350, t269, t270;
        int a1244, b132;
        a1244 = (threadIdx.x + (256*blockIdx.x));
        b132 = ((40*(a1244 / 38)) + ((threadIdx.x + (28*blockIdx.x)) % 38));
        s292 = X[(b132 + 41)];
        s293 = X[(b132 + 1601)];
        s294 = X[(b132 + 1640)];
        s295 = X[(b132 + 1641)];
        s296 = X[(b132 + 1642)];
        s297 = X[(b132 + 1681)];
        s298 = X[(b132 + 3241)];
        s299 = X[(b132 + 64041)];
        s300 = X[(b132 + 65601)];
        s301 = X[(b132 + 65640)];
        s302 = X[(b132 + 65641)];
        s303 = X[(b132 + 65642)];
        s304 = X[(b132 + 65681)];
        s305 = X[(b132 + 67241)];
        s306 = X[(b132 + 128041)];
        s307 = X[(b132 + 129601)];
        s308 = X[(b132 + 129640)];
        s309 = X[(b132 + 129641)];
        s310 = X[(b132 + 129642)];
        s311 = X[(b132 + 129681)];
        s312 = X[(b132 + 131241)];
        s313 = X[(b132 + 192041)];
        s314 = X[(b132 + 193601)];
        s315 = X[(b132 + 193640)];
        s316 = X[(b132 + 193641)];
        s317 = X[(b132 + 193642)];
        s318 = X[(b132 + 193681)];
        s319 = X[(b132 + 195241)];
        s320 = X[(b132 + 256041)];
        s321 = X[(b132 + 257601)];
        s322 = X[(b132 + 257640)];
        s323 = X[(b132 + 257641)];
        s324 = X[(b132 + 257642)];
        s325 = X[(b132 + 257681)];
        s326 = X[(b132 + 259241)];
        s327 = (s299 / s292);
        s328 = (s300 / s293);
        s329 = (s301 / s294);
        s330 = (s302 / s295);
        s331 = (s303 / s296);
        s332 = (s304 / s297);
        s333 = (s305 / s298);
        s334 = (s306 / s292);
        s335 = (s307 / s293);
        s336 = (s308 / s294);
        s337 = (s309 / s295);
        s338 = (s310 / s296);
        s339 = (s311 / s297);
        s340 = (s312 / s298);
        s341 = (s313 / s292);
        s342 = (s314 / s293);
        s343 = (s315 / s294);
        s344 = (s316 / s295);
        s345 = (s317 / s296);
        s346 = (s318 / s297);
        s347 = (s319 / s298);
        a1245 = (gamma1 - 1.0);
        a1246 = (0.5*a1245);
        t269 = (0.041666666666666664*((s292 - (6.0*s295)) + s293 + s294 + s296 + s297 + s298));
        t270 = (s297 - t269);
        s348 = ((s304 - (0.041666666666666664*((s299 - (6.0*s302)) + s300 + s301 + s303 + s304 + s305))) / t270);
        s349 = ((s311 - (0.041666666666666664*((s306 - (6.0*s309)) + s307 + s308 + s310 + s311 + s312))) / t270);
        s350 = ((s318 - (0.041666666666666664*((s313 - (6.0*s316)) + s314 + s315 + s317 + s318 + s319))) / t270);
        P1[a1244] = (t269 + t270);
        P1[(a1244 + 54872)] = ((0.041666666666666664*(((s327 + s328 + s329) - (6.0*s330)) + s331 + s332 + s333)) + s348);
        P1[(a1244 + 109744)] = ((0.041666666666666664*(((s334 + s335 + s336) - (6.0*s337)) + s338 + s339 + s340)) + s349);
        P1[(a1244 + 164616)] = ((0.041666666666666664*(((s341 + s342 + s343) - (6.0*s344)) + s345 + s346 + s347)) + s350);
        P1[(a1244 + 219488)] = ((0.041666666666666664*((((a1245*s320) - ((a1246*((s327*s327)*s292)) + (a1246*((s334*s334)*s292)) + (a1246*((s341*s341)*s292)))) - (6.0*((a1245*s323) - ((a1246*((s330*s330)*s295)) + (a1246*((s337*s337)*s295)) + (a1246*((s344*s344)*s295)))))) + ((a1245*s321) - ((a1246*((s328*s328)*s293)) + (a1246*((s335*s335)*s293)) + (a1246*((s342*s342)*s293)))) + ((a1245*s322) - ((a1246*((s329*s329)*s294)) + (a1246*((s336*s336)*s294)) + (a1246*((s343*s343)*s294)))) + ((a1245*s324) - ((a1246*((s331*s331)*s296)) + (a1246*((s338*s338)*s296)) + (a1246*((s345*s345)*s296)))) + ((a1245*s325) - ((a1246*((s332*s332)*s297)) + (a1246*((s339*s339)*s297)) + (a1246*((s346*s346)*s297)))) + ((a1245*s326) - ((a1246*((s333*s333)*s298)) + (a1246*((s340*s340)*s298)) + (a1246*((s347*s347)*s298)))))) + ((a1245*(s325 - (0.041666666666666664*((s320 - (6.0*s323)) + s321 + s322 + s324 + s325 + s326)))) - ((a1246*((s348*s348)*t270)) + (a1246*((s349*s349)*t270)) + (a1246*((s350*s350)*t270)))));
    }
}

extern "C" __global__ void ker_code1(double gamma1, double *P2, double *P1) {
    if (((((256*blockIdx.x) + threadIdx.x) < 50540))) {
        double a1944, s585, s586, s587, s588, s589, s590, s594, 
                s595, s596, s597, s598, s599, s603, s604, s605, 
                s606, s607, s608, s609, s610, s611, s612, s613, 
                s614, s615, s616, s617, s618, s619, s620, s621, 
                s622, s623, s624, s625, s626, s627, s628, s629, 
                s630, s631, s632, t367, t368, t369, t370, t377, 
                t378, t379, t380, t388, t389, t390, t391, t392, 
                t393, t394, t395, t396, t397, t398, t399, t400, 
                t401, t402, t403, t404, t405, t406, t407, t408, 
                t409, t410, t411, t412, t413, t414, t415, t416, 
                t417, t418;
        int a1886, a1887, a1888, a1889, a1890, a1891, a1892, a1893, 
                a1894, a1895, a1896, a1897, a1898, a1899, a1900, a1901, 
                a1902, a1903, a1904, a1905, a1906, a1907, a1908, a1909, 
                a1910, a1911, a1912, a1913, a1914, a1915, a1916, a1917, 
                a1918, a1919, a1920, a1921, a1923, a1924, a1925, a1926, 
                a1927, a1928, a1929, a1930, a1931, a1932, a1933, a1934, 
                a1935, a1936, a1937, a1938, a1939, a1940, a1941, a1942, 
                a1943, a1945, a1946, a1947, a1948, a1949, a1950, a1951, 
                a1952, a1953, a1954, a1955, a1956, a1957, a1958, a1959, 
                a1960, a1961, a1962, a1963, a1964, a1965, a1966, a1967, 
                a1968, a1969, a1970, a1971, a1972, a1973, a1974, a1975, 
                a1976, a1977, a1978, a1979, a1980, a1981, a1982, a1983, 
                a1984, a1985, a1986, a1987, a1988, a1989, a1990, a1991, 
                a1992, a1993, a1994, a1995, a1996, a1997, a1998, a1999, 
                a2000, a2001;
        a1943 = ((256*blockIdx.x) + threadIdx.x);
        t392 = (0.5*((((0.58333333333333337*P1[(a1943 + 166060)]) - (0.083333333333333329*P1[(a1943 + 164616)])) + (0.58333333333333337*P1[(a1943 + 167504)])) - (0.083333333333333329*P1[(a1943 + 168948)])));
        t393 = (0.5*((((0.58333333333333337*P1[(a1943 + 220932)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 222376)])) - (0.083333333333333329*P1[(a1943 + 223820)])));
        t394 = (0.5*((((0.58333333333333337*P1[(a1943 + 1444)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 2888)])) - (0.083333333333333329*P1[(a1943 + 4332)])));
        t395 = (t394 + t394);
        t396 = (t393 + t393);
        a1944 = sqrt(gamma1);
        s609 = ((a1944*sqrt(t396))*sqrt((1 / t395)));
        s610 = t396;
        if ((((t392 + t392) > 0))) {
            if ((((s609 - (t392 + t392)) > 0))) {
                t397 = ((s610 + (0.083333333333333329*P1[(a1943 + 223820)])) - (((0.58333333333333337*P1[(a1943 + 220932)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 222376)])));
                s611 = (s609*s609);
                s612 = (1 / s611);
                s613 = (t397*s612);
                t398 = (s613 + ((((0.58333333333333337*P1[(a1943 + 1444)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 2888)])) - (0.083333333333333329*P1[(a1943 + 4332)])));
                P2[a1943] = t398;
                a1945 = (a1943 + 50540);
                P2[a1945] = ((((0.58333333333333337*P1[(a1943 + 56316)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 57760)])) - (0.083333333333333329*P1[(a1943 + 59204)]));
                a1946 = (a1943 + 101080);
                P2[a1946] = ((((0.58333333333333337*P1[(a1943 + 111188)]) - (0.083333333333333329*P1[(a1943 + 109744)])) + (0.58333333333333337*P1[(a1943 + 112632)])) - (0.083333333333333329*P1[(a1943 + 114076)]));
                a1947 = (a1943 + 151620);
                P2[a1947] = (t392 + t392);
                a1948 = (a1943 + 202160);
                P2[a1948] = s610;
            } else {
                P2[a1943] = ((((0.58333333333333337*P1[(a1943 + 1444)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 2888)])) - (0.083333333333333329*P1[(a1943 + 4332)]));
                a1949 = (a1943 + 50540);
                P2[a1949] = ((((0.58333333333333337*P1[(a1943 + 56316)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 57760)])) - (0.083333333333333329*P1[(a1943 + 59204)]));
                a1950 = (a1943 + 101080);
                P2[a1950] = ((((0.58333333333333337*P1[(a1943 + 111188)]) - (0.083333333333333329*P1[(a1943 + 109744)])) + (0.58333333333333337*P1[(a1943 + 112632)])) - (0.083333333333333329*P1[(a1943 + 114076)]));
                a1951 = (a1943 + 151620);
                P2[a1951] = ((((0.58333333333333337*P1[(a1943 + 166060)]) - (0.083333333333333329*P1[(a1943 + 164616)])) + (0.58333333333333337*P1[(a1943 + 167504)])) - (0.083333333333333329*P1[(a1943 + 168948)]));
                a1952 = (a1943 + 202160);
                P2[a1952] = ((((0.58333333333333337*P1[(a1943 + 220932)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 222376)])) - (0.083333333333333329*P1[(a1943 + 223820)]));
            }
        } else {
            if ((((s609 + t392 + t392) > 0))) {
                t399 = ((s610 + (0.083333333333333329*P1[(a1943 + 223820)])) - (((0.58333333333333337*P1[(a1943 + 220932)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 222376)])));
                s614 = (s609*s609);
                s615 = (1 / s614);
                s616 = (t399*s615);
                t400 = (s616 + ((((0.58333333333333337*P1[(a1943 + 1444)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 2888)])) - (0.083333333333333329*P1[(a1943 + 4332)])));
                P2[a1943] = t400;
                a1953 = (a1943 + 50540);
                P2[a1953] = ((((0.58333333333333337*P1[(a1943 + 56316)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 57760)])) - (0.083333333333333329*P1[(a1943 + 59204)]));
                a1954 = (a1943 + 101080);
                P2[a1954] = ((((0.58333333333333337*P1[(a1943 + 111188)]) - (0.083333333333333329*P1[(a1943 + 109744)])) + (0.58333333333333337*P1[(a1943 + 112632)])) - (0.083333333333333329*P1[(a1943 + 114076)]));
                a1955 = (a1943 + 151620);
                P2[a1955] = (t392 + t392);
                a1956 = (a1943 + 202160);
                P2[a1956] = s610;
            } else {
                P2[a1943] = ((((0.58333333333333337*P1[(a1943 + 1444)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 2888)])) - (0.083333333333333329*P1[(a1943 + 4332)]));
                a1957 = (a1943 + 50540);
                P2[a1957] = ((((0.58333333333333337*P1[(a1943 + 56316)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 57760)])) - (0.083333333333333329*P1[(a1943 + 59204)]));
                a1958 = (a1943 + 101080);
                P2[a1958] = ((((0.58333333333333337*P1[(a1943 + 111188)]) - (0.083333333333333329*P1[(a1943 + 109744)])) + (0.58333333333333337*P1[(a1943 + 112632)])) - (0.083333333333333329*P1[(a1943 + 114076)]));
                a1959 = (a1943 + 151620);
                P2[a1959] = ((((0.58333333333333337*P1[(a1943 + 166060)]) - (0.083333333333333329*P1[(a1943 + 164616)])) + (0.58333333333333337*P1[(a1943 + 167504)])) - (0.083333333333333329*P1[(a1943 + 168948)]));
                a1960 = (a1943 + 202160);
                P2[a1960] = ((((0.58333333333333337*P1[(a1943 + 220932)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 222376)])) - (0.083333333333333329*P1[(a1943 + 223820)]));
            }
        }
        t401 = (0.5*((((0.58333333333333337*P1[(a1943 + 109782)]) - (0.083333333333333329*P1[(a1943 + 109744)])) + (0.58333333333333337*P1[(a1943 + 109820)])) - (0.083333333333333329*P1[(a1943 + 109858)])));
        t402 = (0.5*((((0.58333333333333337*P1[(a1943 + 219526)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 219564)])) - (0.083333333333333329*P1[(a1943 + 219602)])));
        t403 = (0.5*((((0.58333333333333337*P1[(a1943 + 38)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 76)])) - (0.083333333333333329*P1[(a1943 + 114)])));
        t404 = (t403 + t403);
        t405 = (t402 + t402);
        s617 = ((a1944*sqrt(t405))*sqrt((1 / t404)));
        s618 = t405;
        if ((((t401 + t401) > 0))) {
            if ((((s617 - (t401 + t401)) > 0))) {
                t406 = ((s618 + (0.083333333333333329*P1[(a1943 + 219602)])) - (((0.58333333333333337*P1[(a1943 + 219526)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 219564)])));
                s619 = (s617*s617);
                s620 = (1 / s619);
                s621 = (t406*s620);
                t407 = (s621 + ((((0.58333333333333337*P1[(a1943 + 38)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 76)])) - (0.083333333333333329*P1[(a1943 + 114)])));
                a1961 = (a1943 + 252700);
                P2[a1961] = t407;
                a1962 = (a1943 + 303240);
                P2[a1962] = ((((0.58333333333333337*P1[(a1943 + 54910)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 54948)])) - (0.083333333333333329*P1[(a1943 + 54986)]));
                a1963 = (a1943 + 353780);
                P2[a1963] = (t401 + t401);
                a1964 = (a1943 + 404320);
                P2[a1964] = ((((0.58333333333333337*P1[(a1943 + 164654)]) - (0.083333333333333329*P1[(a1943 + 164616)])) + (0.58333333333333337*P1[(a1943 + 164692)])) - (0.083333333333333329*P1[(a1943 + 164730)]));
                a1965 = (a1943 + 454860);
                P2[a1965] = s618;
            } else {
                a1966 = (a1943 + 252700);
                P2[a1966] = ((((0.58333333333333337*P1[(a1943 + 38)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 76)])) - (0.083333333333333329*P1[(a1943 + 114)]));
                a1967 = (a1943 + 303240);
                P2[a1967] = ((((0.58333333333333337*P1[(a1943 + 54910)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 54948)])) - (0.083333333333333329*P1[(a1943 + 54986)]));
                a1968 = (a1943 + 353780);
                P2[a1968] = ((((0.58333333333333337*P1[(a1943 + 109782)]) - (0.083333333333333329*P1[(a1943 + 109744)])) + (0.58333333333333337*P1[(a1943 + 109820)])) - (0.083333333333333329*P1[(a1943 + 109858)]));
                a1969 = (a1943 + 404320);
                P2[a1969] = ((((0.58333333333333337*P1[(a1943 + 164654)]) - (0.083333333333333329*P1[(a1943 + 164616)])) + (0.58333333333333337*P1[(a1943 + 164692)])) - (0.083333333333333329*P1[(a1943 + 164730)]));
                a1970 = (a1943 + 454860);
                P2[a1970] = ((((0.58333333333333337*P1[(a1943 + 219526)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 219564)])) - (0.083333333333333329*P1[(a1943 + 219602)]));
            }
        } else {
            if ((((s617 + t401 + t401) > 0))) {
                t408 = ((s618 + (0.083333333333333329*P1[(a1943 + 219602)])) - (((0.58333333333333337*P1[(a1943 + 219526)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 219564)])));
                s622 = (s617*s617);
                s623 = (1 / s622);
                s624 = (t408*s623);
                t409 = (s624 + ((((0.58333333333333337*P1[(a1943 + 38)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 76)])) - (0.083333333333333329*P1[(a1943 + 114)])));
                a1971 = (a1943 + 252700);
                P2[a1971] = t409;
                a1972 = (a1943 + 303240);
                P2[a1972] = ((((0.58333333333333337*P1[(a1943 + 54910)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 54948)])) - (0.083333333333333329*P1[(a1943 + 54986)]));
                a1973 = (a1943 + 353780);
                P2[a1973] = (t401 + t401);
                a1974 = (a1943 + 404320);
                P2[a1974] = ((((0.58333333333333337*P1[(a1943 + 164654)]) - (0.083333333333333329*P1[(a1943 + 164616)])) + (0.58333333333333337*P1[(a1943 + 164692)])) - (0.083333333333333329*P1[(a1943 + 164730)]));
                a1975 = (a1943 + 454860);
                P2[a1975] = s618;
            } else {
                a1976 = (a1943 + 252700);
                P2[a1976] = ((((0.58333333333333337*P1[(a1943 + 38)]) - (0.083333333333333329*P1[a1943])) + (0.58333333333333337*P1[(a1943 + 76)])) - (0.083333333333333329*P1[(a1943 + 114)]));
                a1977 = (a1943 + 303240);
                P2[a1977] = ((((0.58333333333333337*P1[(a1943 + 54910)]) - (0.083333333333333329*P1[(a1943 + 54872)])) + (0.58333333333333337*P1[(a1943 + 54948)])) - (0.083333333333333329*P1[(a1943 + 54986)]));
                a1978 = (a1943 + 353780);
                P2[a1978] = ((((0.58333333333333337*P1[(a1943 + 109782)]) - (0.083333333333333329*P1[(a1943 + 109744)])) + (0.58333333333333337*P1[(a1943 + 109820)])) - (0.083333333333333329*P1[(a1943 + 109858)]));
                a1979 = (a1943 + 404320);
                P2[a1979] = ((((0.58333333333333337*P1[(a1943 + 164654)]) - (0.083333333333333329*P1[(a1943 + 164616)])) + (0.58333333333333337*P1[(a1943 + 164692)])) - (0.083333333333333329*P1[(a1943 + 164730)]));
                a1980 = (a1943 + 454860);
                P2[a1980] = ((((0.58333333333333337*P1[(a1943 + 219526)]) - (0.083333333333333329*P1[(a1943 + 219488)])) + (0.58333333333333337*P1[(a1943 + 219564)])) - (0.083333333333333329*P1[(a1943 + 219602)]));
            }
        }
        a1981 = ((38*(a1943 / 38)) + (((28*blockIdx.x) + threadIdx.x) % 38));
        t410 = (0.5*((((0.58333333333333337*P1[(a1981 + 54873)]) - (0.083333333333333329*P1[(a1981 + 54872)])) + (0.58333333333333337*P1[(a1981 + 54874)])) - (0.083333333333333329*P1[(a1981 + 54875)])));
        t411 = (0.5*((((0.58333333333333337*P1[(a1981 + 219489)]) - (0.083333333333333329*P1[(a1981 + 219488)])) + (0.58333333333333337*P1[(a1981 + 219490)])) - (0.083333333333333329*P1[(a1981 + 219491)])));
        t412 = (0.5*((((0.58333333333333337*P1[(a1981 + 1)]) - (0.083333333333333329*P1[a1981])) + (0.58333333333333337*P1[(a1981 + 2)])) - (0.083333333333333329*P1[(a1981 + 3)])));
        t413 = (t412 + t412);
        t414 = (t411 + t411);
        s625 = ((a1944*sqrt(t414))*sqrt((1 / t413)));
        s626 = t414;
        if ((((t410 + t410) > 0))) {
            if ((((s625 - (t410 + t410)) > 0))) {
                t415 = ((s626 + (0.083333333333333329*P1[(a1981 + 219491)])) - (((0.58333333333333337*P1[(a1981 + 219489)]) - (0.083333333333333329*P1[(a1981 + 219488)])) + (0.58333333333333337*P1[(a1981 + 219490)])));
                s627 = (s625*s625);
                s628 = (1 / s627);
                s629 = (t415*s628);
                t416 = (s629 + ((((0.58333333333333337*P1[(a1981 + 1)]) - (0.083333333333333329*P1[a1981])) + (0.58333333333333337*P1[(a1981 + 2)])) - (0.083333333333333329*P1[(a1981 + 3)])));
                a1982 = (a1943 + 505400);
                P2[a1982] = t416;
                a1983 = (a1943 + 555940);
                P2[a1983] = (t410 + t410);
                a1984 = (a1943 + 606480);
                P2[a1984] = ((((0.58333333333333337*P1[(a1981 + 109745)]) - (0.083333333333333329*P1[(a1981 + 109744)])) + (0.58333333333333337*P1[(a1981 + 109746)])) - (0.083333333333333329*P1[(a1981 + 109747)]));
                a1985 = (a1943 + 657020);
                P2[a1985] = ((((0.58333333333333337*P1[(a1981 + 164617)]) - (0.083333333333333329*P1[(a1981 + 164616)])) + (0.58333333333333337*P1[(a1981 + 164618)])) - (0.083333333333333329*P1[(a1981 + 164619)]));
                a1986 = (a1943 + 707560);
                P2[a1986] = s626;
            } else {
                a1987 = (a1943 + 505400);
                P2[a1987] = ((((0.58333333333333337*P1[(a1981 + 1)]) - (0.083333333333333329*P1[a1981])) + (0.58333333333333337*P1[(a1981 + 2)])) - (0.083333333333333329*P1[(a1981 + 3)]));
                a1988 = (a1943 + 555940);
                P2[a1988] = ((((0.58333333333333337*P1[(a1981 + 54873)]) - (0.083333333333333329*P1[(a1981 + 54872)])) + (0.58333333333333337*P1[(a1981 + 54874)])) - (0.083333333333333329*P1[(a1981 + 54875)]));
                a1989 = (a1943 + 606480);
                P2[a1989] = ((((0.58333333333333337*P1[(a1981 + 109745)]) - (0.083333333333333329*P1[(a1981 + 109744)])) + (0.58333333333333337*P1[(a1981 + 109746)])) - (0.083333333333333329*P1[(a1981 + 109747)]));
                a1990 = (a1943 + 657020);
                P2[a1990] = ((((0.58333333333333337*P1[(a1981 + 164617)]) - (0.083333333333333329*P1[(a1981 + 164616)])) + (0.58333333333333337*P1[(a1981 + 164618)])) - (0.083333333333333329*P1[(a1981 + 164619)]));
                a1991 = (a1943 + 707560);
                P2[a1991] = ((((0.58333333333333337*P1[(a1981 + 219489)]) - (0.083333333333333329*P1[(a1981 + 219488)])) + (0.58333333333333337*P1[(a1981 + 219490)])) - (0.083333333333333329*P1[(a1981 + 219491)]));
            }
        } else {
            if ((((s625 + t410 + t410) > 0))) {
                t417 = ((s626 + (0.083333333333333329*P1[(a1981 + 219491)])) - (((0.58333333333333337*P1[(a1981 + 219489)]) - (0.083333333333333329*P1[(a1981 + 219488)])) + (0.58333333333333337*P1[(a1981 + 219490)])));
                s630 = (s625*s625);
                s631 = (1 / s630);
                s632 = (t417*s631);
                t418 = (s632 + ((((0.58333333333333337*P1[(a1981 + 1)]) - (0.083333333333333329*P1[a1981])) + (0.58333333333333337*P1[(a1981 + 2)])) - (0.083333333333333329*P1[(a1981 + 3)])));
                a1992 = (a1943 + 505400);
                P2[a1992] = t418;
                a1993 = (a1943 + 555940);
                P2[a1993] = (t410 + t410);
                a1994 = (a1943 + 606480);
                P2[a1994] = ((((0.58333333333333337*P1[(a1981 + 109745)]) - (0.083333333333333329*P1[(a1981 + 109744)])) + (0.58333333333333337*P1[(a1981 + 109746)])) - (0.083333333333333329*P1[(a1981 + 109747)]));
                a1995 = (a1943 + 657020);
                P2[a1995] = ((((0.58333333333333337*P1[(a1981 + 164617)]) - (0.083333333333333329*P1[(a1981 + 164616)])) + (0.58333333333333337*P1[(a1981 + 164618)])) - (0.083333333333333329*P1[(a1981 + 164619)]));
                a1996 = (a1943 + 707560);
                P2[a1996] = s626;
            } else {
                a1997 = (a1943 + 505400);
                P2[a1997] = ((((0.58333333333333337*P1[(a1981 + 1)]) - (0.083333333333333329*P1[a1981])) + (0.58333333333333337*P1[(a1981 + 2)])) - (0.083333333333333329*P1[(a1981 + 3)]));
                a1998 = (a1943 + 555940);
                P2[a1998] = ((((0.58333333333333337*P1[(a1981 + 54873)]) - (0.083333333333333329*P1[(a1981 + 54872)])) + (0.58333333333333337*P1[(a1981 + 54874)])) - (0.083333333333333329*P1[(a1981 + 54875)]));
                a1999 = (a1943 + 606480);
                P2[a1999] = ((((0.58333333333333337*P1[(a1981 + 109745)]) - (0.083333333333333329*P1[(a1981 + 109744)])) + (0.58333333333333337*P1[(a1981 + 109746)])) - (0.083333333333333329*P1[(a1981 + 109747)]));
                a2000 = (a1943 + 657020);
                P2[a2000] = ((((0.58333333333333337*P1[(a1981 + 164617)]) - (0.083333333333333329*P1[(a1981 + 164616)])) + (0.58333333333333337*P1[(a1981 + 164618)])) - (0.083333333333333329*P1[(a1981 + 164619)]));
                a2001 = (a1943 + 707560);
                P2[a2001] = ((((0.58333333333333337*P1[(a1981 + 219489)]) - (0.083333333333333329*P1[(a1981 + 219488)])) + (0.58333333333333337*P1[(a1981 + 219490)])) - (0.083333333333333329*P1[(a1981 + 219491)]));
            }
        }
    }
}

extern "C" __global__ void ker_code2(double gamma1, double *P2, double *P1) {
    if (((((256*blockIdx.x) + threadIdx.x) < 45360))) {
        double a3215, s1346, s1347, s1348, s1349, s1350, s1351, s1352, 
                s1353, s1354, s1355, s1356, s1357, s1358, s1359, s1360, 
                s1361, s1362, s1363, s1364, s1365, s1366, s1367, s1368, 
                s1369, s1370, s1371, s1372, s1373, s1374, s1375, s1376, 
                s1377, s1378, s1379, s1380, s1381, s1382, s1383, s1384, 
                s1385, s1386, s1387, s1388, s1389, s1390, s1391, s1392, 
                s1393, s1394, s1395, s1396, s1397, s1398, s1399, s1400, 
                s1401, s1402, s1403, s1404, s1405, s1406, s1407, s1408, 
                s1409, s1410, s1411, s1412, s1413, s1414, s1415, s1416, 
                s1417, s1418, s1419, s1420, s1421, s1422, s1423, s1424, 
                s1425, s1426, s1427, s1428, s1429, s1430, s1431, s1432, 
                s1433, s1434, s1435, s1436, s1437, s1438, t695, t696, 
                t697, t698, t699, t700, t701, t702, t703, t704, 
                t705, t706;
        int a3214, a3216, a3217, a3218, b615, b616, b617, b618, 
                b619;
        a3214 = ((256*blockIdx.x) + threadIdx.x);
        b615 = ((38*(a3214 / 36)) + (((4*blockIdx.x) + threadIdx.x) % 36));
        s1346 = P2[(b615 + 505401)];
        s1347 = P2[(b615 + 505438)];
        s1348 = P2[(b615 + 505439)];
        s1349 = P2[(b615 + 505440)];
        s1350 = P2[(b615 + 505477)];
        s1351 = P2[(b615 + 555941)];
        s1352 = P2[(b615 + 555978)];
        s1353 = P2[(b615 + 555979)];
        s1354 = P2[(b615 + 555980)];
        s1355 = P2[(b615 + 556017)];
        s1356 = P2[(b615 + 606481)];
        s1357 = P2[(b615 + 606518)];
        s1358 = P2[(b615 + 606519)];
        s1359 = P2[(b615 + 606520)];
        s1360 = P2[(b615 + 606557)];
        s1361 = P2[(b615 + 657021)];
        s1362 = P2[(b615 + 657058)];
        s1363 = P2[(b615 + 657059)];
        s1364 = P2[(b615 + 657060)];
        s1365 = P2[(b615 + 657097)];
        s1366 = P2[(b615 + 707561)];
        s1367 = P2[(b615 + 707598)];
        s1368 = P2[(b615 + 707599)];
        s1369 = P2[(b615 + 707600)];
        s1370 = P2[(b615 + 707637)];
        s1371 = (s1346*s1351);
        s1372 = (s1347*s1352);
        s1373 = (s1348*s1353);
        s1374 = (s1349*s1354);
        s1375 = (s1350*s1355);
        a3215 = (gamma1 / (gamma1 - 1.0));
        a3216 = (a3214 % 1296);
        b616 = ((38*(a3216 / 36)) + (a3216 % 36) + (1444*(a3214 / 1296)));
        t695 = (P2[(b616 + 555979)] - (0.041666666666666664*((s1351 - (4.0*s1353)) + s1352 + s1354 + s1355)));
        t696 = (P2[(b616 + 606519)] - (0.041666666666666664*((s1356 - (4.0*s1358)) + s1357 + s1359 + s1360)));
        t697 = (P2[(b616 + 657059)] - (0.041666666666666664*((s1361 - (4.0*s1363)) + s1362 + s1364 + s1365)));
        t698 = (P2[(b616 + 707599)] - (0.041666666666666664*((s1366 - (4.0*s1368)) + s1367 + s1369 + s1370)));
        s1376 = ((P2[(b616 + 505439)] - (0.041666666666666664*((s1346 - (4.0*s1348)) + s1347 + s1349 + s1350)))*t697);
        P1[a3214] = ((0.041666666666666664*((((s1346*s1361) + (s1347*s1362)) - (4.0*(s1348*s1363))) + (s1349*s1364) + (s1350*s1365))) - s1376);
        P1[(a3214 + 45360)] = ((0.041666666666666664*((((s1371*s1351) + (s1372*s1352)) - (4.0*(s1373*s1353))) + (s1374*s1354) + (s1375*s1355))) - (s1376*t695));
        P1[(a3214 + 90720)] = ((0.041666666666666664*((((s1371*s1356) + (s1372*s1357)) - (4.0*(s1373*s1358))) + (s1374*s1359) + (s1375*s1360))) - (s1376*t696));
        P1[(a3214 + 136080)] = ((0.041666666666666664*((((s1371*s1361) + s1366 + (s1372*s1362) + s1367) - (4.0*((s1373*s1363) + s1368))) + (s1374*s1364) + s1369 + (s1375*s1365) + s1370)) - ((s1376*t697) + t698));
        P1[(a3214 + 181440)] = ((0.041666666666666664*((((a3215*(s1361*s1366)) + (s1371*(((0.5*s1351)*s1351) + ((0.5*s1356)*s1356) + ((0.5*s1361)*s1361))) + (a3215*(s1362*s1367)) + (s1372*(((0.5*s1352)*s1352) + ((0.5*s1357)*s1357) + ((0.5*s1362)*s1362)))) - (4.0*((a3215*(s1363*s1368)) + (s1373*(((0.5*s1353)*s1353) + ((0.5*s1358)*s1358) + ((0.5*s1363)*s1363)))))) + (a3215*(s1364*s1369)) + (s1374*(((0.5*s1354)*s1354) + ((0.5*s1359)*s1359) + ((0.5*s1364)*s1364))) + (a3215*(s1365*s1370)) + (s1375*(((0.5*s1355)*s1355) + ((0.5*s1360)*s1360) + ((0.5*s1365)*s1365))))) - ((a3215*(t697*t698)) + (s1376*(((0.5*t695)*t695) + ((0.5*t696)*t696) + ((0.5*t697)*t697)))));
        s1377 = P2[(b615 + 252701)];
        s1378 = P2[(b615 + 254030)];
        s1379 = P2[(b615 + 254031)];
        s1380 = P2[(b615 + 254032)];
        s1381 = P2[(b615 + 255361)];
        s1382 = P2[(b615 + 303241)];
        s1383 = P2[(b615 + 304570)];
        s1384 = P2[(b615 + 304571)];
        s1385 = P2[(b615 + 304572)];
        s1386 = P2[(b615 + 305901)];
        s1387 = P2[(b615 + 353781)];
        s1388 = P2[(b615 + 355110)];
        s1389 = P2[(b615 + 355111)];
        s1390 = P2[(b615 + 355112)];
        s1391 = P2[(b615 + 356441)];
        s1392 = P2[(b615 + 404321)];
        s1393 = P2[(b615 + 405650)];
        s1394 = P2[(b615 + 405651)];
        s1395 = P2[(b615 + 405652)];
        s1396 = P2[(b615 + 406981)];
        s1397 = P2[(b615 + 454861)];
        s1398 = P2[(b615 + 456190)];
        s1399 = P2[(b615 + 456191)];
        s1400 = P2[(b615 + 456192)];
        s1401 = P2[(b615 + 457521)];
        s1402 = (s1377*s1382);
        s1403 = (s1378*s1383);
        s1404 = (s1379*s1384);
        s1405 = (s1380*s1385);
        s1406 = (s1381*s1386);
        a3217 = (a3214 % 1260);
        a3218 = (1330*(a3214 / 1260));
        b617 = ((38*(a3217 / 36)) + (a3217 % 36) + a3218);
        t699 = (P2[(b617 + 304571)] - (0.041666666666666664*((s1382 - (4.0*s1384)) + s1383 + s1385 + s1386)));
        t700 = (P2[(b617 + 355111)] - (0.041666666666666664*((s1387 - (4.0*s1389)) + s1388 + s1390 + s1391)));
        t701 = (P2[(b617 + 405651)] - (0.041666666666666664*((s1392 - (4.0*s1394)) + s1393 + s1395 + s1396)));
        t702 = (P2[(b617 + 456191)] - (0.041666666666666664*((s1397 - (4.0*s1399)) + s1398 + s1400 + s1401)));
        s1407 = ((P2[(b617 + 254031)] - (0.041666666666666664*((s1377 - (4.0*s1379)) + s1378 + s1380 + s1381)))*t700);
        P1[(a3214 + 226800)] = ((0.041666666666666664*((((s1377*s1387) + (s1378*s1388)) - (4.0*(s1379*s1389))) + (s1380*s1390) + (s1381*s1391))) - s1407);
        P1[(a3214 + 272160)] = ((0.041666666666666664*((((s1402*s1382) + (s1403*s1383)) - (4.0*(s1404*s1384))) + (s1405*s1385) + (s1406*s1386))) - (s1407*t699));
        P1[(a3214 + 317520)] = ((0.041666666666666664*((((s1402*s1387) + s1397 + (s1403*s1388) + s1398) - (4.0*((s1404*s1389) + s1399))) + (s1405*s1390) + s1400 + (s1406*s1391) + s1401)) - ((s1407*t700) + t702));
        P1[(a3214 + 362880)] = ((0.041666666666666664*((((s1402*s1392) + (s1403*s1393)) - (4.0*(s1404*s1394))) + (s1405*s1395) + (s1406*s1396))) - (s1407*t701));
        P1[(a3214 + 408240)] = ((0.041666666666666664*((((a3215*(s1387*s1397)) + (s1402*(((0.5*s1382)*s1382) + ((0.5*s1387)*s1387) + ((0.5*s1392)*s1392))) + (a3215*(s1388*s1398)) + (s1403*(((0.5*s1383)*s1383) + ((0.5*s1388)*s1388) + ((0.5*s1393)*s1393)))) - (4.0*((a3215*(s1389*s1399)) + (s1404*(((0.5*s1384)*s1384) + ((0.5*s1389)*s1389) + ((0.5*s1394)*s1394)))))) + (a3215*(s1390*s1400)) + (s1405*(((0.5*s1385)*s1385) + ((0.5*s1390)*s1390) + ((0.5*s1395)*s1395))) + (a3215*(s1391*s1401)) + (s1406*(((0.5*s1386)*s1386) + ((0.5*s1391)*s1391) + ((0.5*s1396)*s1396))))) - ((a3215*(t700*t702)) + (s1407*(((0.5*t699)*t699) + ((0.5*t700)*t700) + ((0.5*t701)*t701)))));
        b618 = ((35*(a3214 / 35)) + (((11*blockIdx.x) + threadIdx.x) % 35));
        s1408 = P2[(b618 + 35)];
        s1409 = P2[(b618 + 1330)];
        s1410 = P2[(b618 + 1365)];
        s1411 = P2[(b618 + 1400)];
        s1412 = P2[(b618 + 2695)];
        s1413 = P2[(b618 + 50575)];
        s1414 = P2[(b618 + 51870)];
        s1415 = P2[(b618 + 51905)];
        s1416 = P2[(b618 + 51940)];
        s1417 = P2[(b618 + 53235)];
        s1418 = P2[(b618 + 101115)];
        s1419 = P2[(b618 + 102410)];
        s1420 = P2[(b618 + 102445)];
        s1421 = P2[(b618 + 102480)];
        s1422 = P2[(b618 + 103775)];
        s1423 = P2[(b618 + 151655)];
        s1424 = P2[(b618 + 152950)];
        s1425 = P2[(b618 + 152985)];
        s1426 = P2[(b618 + 153020)];
        s1427 = P2[(b618 + 154315)];
        s1428 = P2[(b618 + 202195)];
        s1429 = P2[(b618 + 203490)];
        s1430 = P2[(b618 + 203525)];
        s1431 = P2[(b618 + 203560)];
        s1432 = P2[(b618 + 204855)];
        s1433 = (s1408*s1413);
        s1434 = (s1409*s1414);
        s1435 = (s1410*s1415);
        s1436 = (s1411*s1416);
        s1437 = (s1412*s1417);
        b619 = ((35*(a3217 / 35)) + (a3217 % 35) + a3218);
        t703 = (P2[(b619 + 51905)] - (0.041666666666666664*((s1413 - (4.0*s1415)) + s1414 + s1416 + s1417)));
        t704 = (P2[(b619 + 102445)] - (0.041666666666666664*((s1418 - (4.0*s1420)) + s1419 + s1421 + s1422)));
        t705 = (P2[(b619 + 152985)] - (0.041666666666666664*((s1423 - (4.0*s1425)) + s1424 + s1426 + s1427)));
        t706 = (P2[(b619 + 203525)] - (0.041666666666666664*((s1428 - (4.0*s1430)) + s1429 + s1431 + s1432)));
        s1438 = ((P2[(b619 + 1365)] - (0.041666666666666664*((s1408 - (4.0*s1410)) + s1409 + s1411 + s1412)))*t703);
        P1[(a3214 + 453600)] = ((0.041666666666666664*(((s1433 + s1434) - (4.0*s1435)) + s1436 + s1437)) - s1438);
        P1[(a3214 + 498960)] = ((0.041666666666666664*((((s1433*s1413) + s1428 + (s1434*s1414) + s1429) - (4.0*((s1435*s1415) + s1430))) + (s1436*s1416) + s1431 + (s1437*s1417) + s1432)) - ((s1438*t703) + t706));
        P1[(a3214 + 544320)] = ((0.041666666666666664*((((s1433*s1418) + (s1434*s1419)) - (4.0*(s1435*s1420))) + (s1436*s1421) + (s1437*s1422))) - (s1438*t704));
        P1[(a3214 + 589680)] = ((0.041666666666666664*((((s1433*s1423) + (s1434*s1424)) - (4.0*(s1435*s1425))) + (s1436*s1426) + (s1437*s1427))) - (s1438*t705));
        P1[(a3214 + 635040)] = ((0.041666666666666664*((((a3215*(s1413*s1428)) + (s1433*(((0.5*s1413)*s1413) + ((0.5*s1418)*s1418) + ((0.5*s1423)*s1423))) + (a3215*(s1414*s1429)) + (s1434*(((0.5*s1414)*s1414) + ((0.5*s1419)*s1419) + ((0.5*s1424)*s1424)))) - (4.0*((a3215*(s1415*s1430)) + (s1435*(((0.5*s1415)*s1415) + ((0.5*s1420)*s1420) + ((0.5*s1425)*s1425)))))) + (a3215*(s1416*s1431)) + (s1436*(((0.5*s1416)*s1416) + ((0.5*s1421)*s1421) + ((0.5*s1426)*s1426))) + (a3215*(s1417*s1432)) + (s1437*(((0.5*s1417)*s1417) + ((0.5*s1422)*s1422) + ((0.5*s1427)*s1427))))) - ((a3215*(t703*t706)) + (s1438*(((0.5*t703)*t703) + ((0.5*t704)*t704) + ((0.5*t705)*t705)))));
    }
}

extern "C" __global__ void ker_code3(double *P2, double *P1) {
    if (((((256*blockIdx.x) + threadIdx.x) < 32768))) {
        int a3672, a3673, a3674, a3675, a3676, a3677, b670, b671, 
                b672;
        a3672 = ((256*blockIdx.x) + threadIdx.x);
        a3673 = (a3672 / 32);
        a3674 = (threadIdx.x % 32);
        a3675 = (((36*a3673) + a3674) % 1152);
        a3676 = (a3672 / 1024);
        b670 = (a3675 + (1296*a3676));
        P2[a3672] = (P1[(b670 + 457560)] - P1[(b670 + 456264)]);
        P2[(a3672 + 32768)] = (P1[(b670 + 502920)] - P1[(b670 + 501624)]);
        P2[(a3672 + 65536)] = (P1[(b670 + 548280)] - P1[(b670 + 546984)]);
        P2[(a3672 + 98304)] = (P1[(b670 + 593640)] - P1[(b670 + 592344)]);
        P2[(a3672 + 131072)] = (P1[(b670 + 639000)] - P1[(b670 + 637704)]);
        a3677 = (1260*a3676);
        b671 = (a3675 + a3677);
        P2[(a3672 + 163840)] = (P1[(b671 + 229428)] - P1[(b671 + 229392)]);
        P2[(a3672 + 196608)] = (P1[(b671 + 274788)] - P1[(b671 + 274752)]);
        P2[(a3672 + 229376)] = (P1[(b671 + 320148)] - P1[(b671 + 320112)]);
        P2[(a3672 + 262144)] = (P1[(b671 + 365508)] - P1[(b671 + 365472)]);
        P2[(a3672 + 294912)] = (P1[(b671 + 410868)] - P1[(b671 + 410832)]);
        b672 = ((((35*a3673) + a3674) % 1120) + a3677);
        P2[(a3672 + 327680)] = (P1[(b672 + 2592)] - P1[(b672 + 2591)]);
        P2[(a3672 + 360448)] = (P1[(b672 + 47952)] - P1[(b672 + 47951)]);
        P2[(a3672 + 393216)] = (P1[(b672 + 93312)] - P1[(b672 + 93311)]);
        P2[(a3672 + 425984)] = (P1[(b672 + 138672)] - P1[(b672 + 138671)]);
        P2[(a3672 + 458752)] = (P1[(b672 + 184032)] - P1[(b672 + 184031)]);
    }
}

extern "C" __global__ void ker_code4(double *Y, double a_scale1, double dx1, double *P2) {
    if (((((256*blockIdx.x) + threadIdx.x) < 32768))) {
        double a3747;
        int a3746, a3748, a3749, a3750, a3751;
        a3746 = ((256*blockIdx.x) + threadIdx.x);
        a3747 = (a_scale1 / dx1);
        Y[a3746] = (a3747*(P2[a3746] + P2[(a3746 + 163840)] + P2[(a3746 + 327680)]));
        a3748 = (a3746 + 32768);
        Y[a3748] = (a3747*(P2[a3748] + P2[(a3746 + 196608)] + P2[(a3746 + 360448)]));
        a3749 = (a3746 + 65536);
        Y[a3749] = (a3747*(P2[a3749] + P2[(a3746 + 229376)] + P2[(a3746 + 393216)]));
        a3750 = (a3746 + 98304);
        Y[a3750] = (a3747*(P2[a3750] + P2[(a3746 + 262144)] + P2[(a3746 + 425984)]));
        a3751 = (a3746 + 131072);
        Y[a3751] = (a3747*(P2[a3751] + P2[(a3746 + 294912)] + P2[(a3746 + 458752)]));
    }
}
