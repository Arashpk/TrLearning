package org.tensorflow.lite.examples.transfer;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;


public class TensorFlowLiteClassifier implements Classifier {

    private static float[] baseVotingArray= new float[6];
    private Interpreter tfLite;
    private String name;
    private List<String> labels;


    public static float[] getBaseVotingArray() {
        return baseVotingArray;
    }

    private static List<String> readLabels(AssetManager am, String fileName) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(am.open(fileName)));

        String line;
        List<String> labels = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }

        br.close();
        return labels;
    }

    public static TensorFlowLiteClassifier create(AssetManager assetManager, String name,
                                                  String modelPath, String labelFile) throws IOException {

        TensorFlowLiteClassifier c = new TensorFlowLiteClassifier();
        c.name = name;

        // read labels for label file
        c.labels = readLabels(assetManager, labelFile);

        // set its model path and where the raw asset files are
        c.tfLite = new Interpreter(loadModelFile(assetManager, modelPath));
        int numClasses = 10;

        return c;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public Classification recognize(final float[] pixels) {
        //float [][][][] pixels2d = new float[1][28][28][1];
        /*for(int i=0; i<28;i++) {
            for (int j = 0; j < 28; j++) {
                pixels2d[0][i][j][0] = pixels[(i * 28) + j];
            }
        }*/
        float[][][][] Signal3Axis = new float[1][1][240][1];
        int j = 0;
        for (int i = 0; i < 240; i = i + 1) {
            Signal3Axis[0][0][i][0] = pixels[i];
        }
        float[][] labelProb = new float[1][labels.size()];

        tfLite.run(Signal3Axis, labelProb);

        // Post-processing
        Classification ans = new Classification();
        int max_i = 0;
        float max_p = -1.0F;
        for (int i = 0; i < labels.size(); ++i) {
            baseVotingArray[i] = labelProb[0][i];
            if (labelProb[0][i] > max_p) {
                max_i = i;
                max_p = labelProb[0][i];
            }
        }

        ans.update(max_p, labels.get(max_i));
        return ans;
    }

    // memory-map the model file in assets
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}