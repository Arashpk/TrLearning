package org.tensorflow.lite.examples.transfer;

import com.paramsen.noise.Noise;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;

public class functions {
    private int windowLength, Fs;
    private int FEATURE_LENGTH, SAMPLE_LENGTH;
    private float[] acc_X_array;
    private float[] acc_Y_array;
    private float[] acc_Z_array;

    private double[] realTimeFeature = new double[12];
    private double acc_X_DC = 0, acc_Y_DC = 0, acc_Z_DC = 0;
    private double acc_X_std = 0, acc_Y_std = 0, acc_Z_std = 0;
    private double acc_X_variance, acc_Y_variance, acc_Z_variance;
    private double[][] trainedSampleTable;
    private String[] categoryArray;
    private List<String> ActivityTypes = Arrays.asList("downstairs", "jogging", "running", "standing", "upstairs", "walking");
    private List<Integer> votingArrayEU = new ArrayList<Integer>();
    private List<Integer> votingArrayMAN = new ArrayList<Integer>();


    public void SetVars(int windowLength_, int Fs_,
                        int FEATURE_LENGTH_, int SAMPLE_LENGTH_,
                        float acc_X_array_[], float acc_Y_array_[], float acc_Z_array_[],
                        double[][] trainedSampleTable_,
                        String[] categoryArray_) {
        windowLength = windowLength_;
        Fs = Fs_;
        FEATURE_LENGTH = FEATURE_LENGTH_;
        SAMPLE_LENGTH = SAMPLE_LENGTH_;
        acc_X_array = acc_X_array_;
        acc_Y_array = acc_Y_array_;
        acc_Z_array = acc_Z_array_;
        trainedSampleTable = trainedSampleTable_;
        categoryArray = categoryArray_;
    }

    public void FeatureExtractor() {
        Noise noise = Noise.real(windowLength);
        float[] fft_acc_X = new float[windowLength + 2]; //real output length equals src+2
        float[] fft_acc_Y = new float[windowLength + 2]; //real output length equals src+2
        float[] fft_acc_Z = new float[windowLength + 2]; //real output length equals src+2

        float[] fft_X = noise.fft(acc_X_array, fft_acc_X);
        float[] fft_Y = noise.fft(acc_Y_array, fft_acc_Y);
        float[] fft_Z = noise.fft(acc_Z_array, fft_acc_Z);
        double[] mag_acc_X = new double[windowLength + 2];
        double[] mag_acc_Y = new double[windowLength + 2];
        double[] mag_acc_Z = new double[windowLength + 2];

        // DC detection
        acc_X_DC = DCvalue(acc_X_array);
        acc_Y_DC = DCvalue(acc_Y_array);
        acc_Z_DC = DCvalue(acc_Z_array);
        // variance
        acc_X_variance = Variance(acc_X_array);
        acc_Y_variance = Variance(acc_Y_array);
        acc_Z_variance = Variance(acc_Z_array);
        // std
        acc_X_std = Math.sqrt(acc_X_variance);
        acc_Y_std = Math.sqrt(acc_Y_variance);
        acc_Z_std = Math.sqrt(acc_Z_variance);

        for (int i = 0; i < fft_X.length / 2; i++) {
            float real_X = fft_X[i * 2];
            float real_Y = fft_Y[i * 2];
            float real_Z = fft_Z[i * 2];

            float imaginary_X = fft_X[i * 2 + 1];
            float imaginary_Y = fft_Y[i * 2 + 1];
            float imaginary_Z = fft_Z[i * 2 + 1];

            mag_acc_X[i] = Math.sqrt(Math.pow(real_X, 2) + Math.pow(imaginary_X, 2)) / windowLength;
            mag_acc_Y[i] = Math.sqrt(Math.pow(real_Y, 2) + Math.pow(imaginary_Y, 2)) / windowLength;
            mag_acc_Z[i] = Math.sqrt(Math.pow(real_Z, 2) + Math.pow(imaginary_Z, 2)) / windowLength;
        }

        double acc_X_Frq_max = 0, acc_X_Mag_max = 0;
        double acc_Y_Frq_max = 0, acc_Y_Mag_max = 0;
        double acc_Z_Frq_max = 0, acc_Z_Mag_max = 0;

        for (int i = 1; i < windowLength / 2; i++) {


            if (mag_acc_X[i] > acc_X_Mag_max) {
                acc_X_Mag_max = mag_acc_X[i];
                acc_X_Frq_max = ((i * Fs) / Double.valueOf(windowLength));
            }
            if (mag_acc_Y[i] > acc_Y_Mag_max) {
                acc_Y_Mag_max = mag_acc_Y[i];
                acc_Y_Frq_max = ((i * Fs) / Double.valueOf(windowLength));
            }
            if (mag_acc_Z[i] > acc_Z_Mag_max) {
                acc_Z_Mag_max = mag_acc_Z[i];
                acc_Z_Frq_max = ((i * Fs) / Double.valueOf(windowLength));
            }
        }
        // Load seperated features in a single array
        realTimeFeature[0] = acc_X_DC;
        realTimeFeature[1] = acc_Y_DC;
        realTimeFeature[2] = acc_Z_DC;
        realTimeFeature[3] = acc_X_Mag_max;
        realTimeFeature[4] = acc_X_Frq_max;
        realTimeFeature[5] = acc_Y_Mag_max;
        realTimeFeature[6] = acc_Y_Frq_max;
        realTimeFeature[7] = acc_Z_Mag_max;
        realTimeFeature[8] = acc_Z_Frq_max;
        realTimeFeature[9] = acc_X_std;
        realTimeFeature[10] = acc_Y_std;
        realTimeFeature[11] = acc_Z_std;

    }

    public String categorize() {

        FeatureExtractor();
        Double[] EuclideanDistanceArray = EuclideanDistance(realTimeFeature, trainedSampleTable, categoryArray);
        Double[] ManhattanDistanceArray = ManhattanDistance(realTimeFeature, trainedSampleTable, categoryArray);
        int K = 5;
        String Activity = kNN(EuclideanDistanceArray, ManhattanDistanceArray, categoryArray, K);
        return (Activity);
    }

    private Double[] EuclideanDistance(double[] realTimeFeature, double[][] trainedSampleTable, String[] categoryArray) {
        int featureLength = realTimeFeature.length;
        int categoryLength = categoryArray.length;
        Double[] EuclideanDistanceArray = new Double[categoryLength];

        for (int row = 0; row < categoryLength; row++) {
            //calculate the Euclidean distance
            double EuclideanDistance = 0;
            for (int i = 0; i < featureLength; i++) {
                EuclideanDistance = EuclideanDistance + Math.pow(realTimeFeature[i] - trainedSampleTable[row][i], 2);
            }
            EuclideanDistance = Math.sqrt(EuclideanDistance);
            EuclideanDistanceArray[row] = EuclideanDistance;

        }
        return EuclideanDistanceArray;
    }

    private Double[] ManhattanDistance(double[] realTimeFeature, double[][] trainedSampleTable, String[] categoryArray) {
        int featureLength = realTimeFeature.length;
        int categoryLength = categoryArray.length;
        Double[] ManhattanDistanceArray = new Double[categoryLength];

        for (int row = 0; row < categoryLength; row++) {
            //calculate the Manhattan distance
            double ManhattanDistance = 0;
            for (int i = 0; i < featureLength; i++) {
                ManhattanDistance = ManhattanDistance + Math.abs(realTimeFeature[i] - trainedSampleTable[row][i]);
            }
            ManhattanDistanceArray[row] = ManhattanDistance;
        }
        return ManhattanDistanceArray;
    }

    private String kNN(Double[] euclideanDistanceArray, Double[] manhattanDistanceArray, String[] categoryArray, int k) {

        ArrayList<Double> euclideanDistanceList = new ArrayList<Double>(Arrays.asList(euclideanDistanceArray));
        ArrayList<Double> manhattanDistanceList = new ArrayList<Double>(Arrays.asList(manhattanDistanceArray));

        int featureLength = realTimeFeature.length;
        int categoryLength = categoryArray.length;
        String[] selectedCategoriesEU = new String[k];
        Double[] nearestNeighborsEU = new Double[k];
        Integer[] nearestNeighborsIndexEU = new Integer[k];

        String[] selectedCategoriesMAN = new String[k];
        Double[] nearestNeighborsMAN = new Double[k];
        Integer[] nearestNeighborsIndexMAN = new Integer[k];

        for (int i = 0; i < k; i++) {
            Double min = Collections.min(euclideanDistanceList);
            int minIndex = euclideanDistanceList.indexOf(min);
            nearestNeighborsEU[i] = min;
            nearestNeighborsIndexEU[i] = minIndex;
            selectedCategoriesEU[i] = categoryArray[minIndex];
            euclideanDistanceList.remove(minIndex);
        }
        for (int i = 0; i < k; i++) {
            Double min = Collections.min(manhattanDistanceList);
            int minIndex = manhattanDistanceList.indexOf(min);
            nearestNeighborsMAN[i] = min;
            nearestNeighborsIndexMAN[i] = minIndex;
            selectedCategoriesMAN[i] = categoryArray[minIndex];
            manhattanDistanceList.remove(minIndex);
        }

        ArrayList<String> selectedCategoriesEUList = new ArrayList<String>(Arrays.asList(selectedCategoriesEU));
        ArrayList<String> selectedCategoriesMANList = new ArrayList<String>(Arrays.asList(selectedCategoriesMAN));

        votingArrayEU = new ArrayList<Integer>();
        votingArrayMAN = new ArrayList<Integer>();
        for (int i = 0; i < ActivityTypes.size(); i++) {
            int frqOfActivity = Collections.frequency(selectedCategoriesEUList, ActivityTypes.get(i));
            votingArrayEU.add(i, frqOfActivity);
        }
        for (int i = 0; i < ActivityTypes.size(); i++) {
            int frqOfActivity = Collections.frequency(selectedCategoriesMANList, ActivityTypes.get(i));
            votingArrayMAN.add(i, frqOfActivity);
        }
        //Log.d("Confusion Matrix", ActivityTypes.toString());
        //Log.d("Confusion Matrix", votingArrayEU.toString());
        int finalVoteEUIndex = Collections.max(votingArrayEU);
        String finalVoteEUstring = ActivityTypes.get(votingArrayEU.indexOf(finalVoteEUIndex));

        int finalVoteMANIndex = Collections.max(votingArrayMAN);
        String finalVoteMANstring = ActivityTypes.get(votingArrayMAN.indexOf(finalVoteMANIndex));
        return finalVoteEUstring;
    }

    String fileNAmeGenerator() {
        Calendar calendar = Calendar.getInstance();
        String currentDate = DateFormat.getDateInstance().format(calendar.getTime());
        SimpleDateFormat format = new SimpleDateFormat("HH_mm_ss");
        String time = format.format(calendar.getTime());
        String finalFileName = currentDate + "_" + time + "_" + ".csv";
        return finalFileName;
    }

    List<Integer> getVotingArrayEU() {
        return (votingArrayEU);
    }

    double DCvalue(float[] signalArray) {
        double DC = 0;
        for (int i = 0; i < windowLength; i++) {
            DC = DC + signalArray[i];
        }
        DC = DC / Double.valueOf(windowLength);
        return (DC);
    }

    double Variance(float[] signalArray) {
        double variance = 0;
        for (int i = 0; i < windowLength; i++) {
            variance = variance + Math.pow(acc_X_array[i] - acc_X_DC, 2);
        }
        variance = variance / (Double.valueOf(windowLength) - 1);
        return (variance);
    }
}
