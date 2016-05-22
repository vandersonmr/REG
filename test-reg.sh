#!/bin/bash
TIME=${TIME:-`which time` -f%e}
DATA=${DATA:-simple-syntetic.cut44s.su}
PROG=${PROG:-./build/reg}

echo "Test 1:"
echo

ARGS="4120 -480 1.124 0.005
-0.1     0.1     20 
-0.00143 0.00057 20 
 7.8e-07 9.8e-07 20 
-1.0e-07 1.0e-07 20 
-1.0e-07 1.0e-07 20"
$TIME $PROG $ARGS $DATA

echo
echo "Test 2:"
echo

ARGS="4120 -480 1.94 0.005
-0.00088484  0.00111516 20 
-0.001194    0.000806   20 
 6.4e-07     8.4e-07    20 
 6.0e-10     8.0e-10    20 
 4.61e-08    6.61e-08   20"
$TIME $PROG $ARGS $DATA

echo
echo "Test 3:"
echo

ARGS="4120 -480 2.255 0.005
-0.001147   0.000853  20 
-0.001139   0.000861  20 
 4.396e-07  5.396e-07 20 
 3.002e-07  4.102e-07 20 
-2.101e-07  0.101e-07 20"
$TIME $PROG $ARGS $DATA

