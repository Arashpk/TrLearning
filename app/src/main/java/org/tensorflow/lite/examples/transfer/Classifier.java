package org.tensorflow.lite.examples.transfer;

public interface Classifier {
    String name();

    Classification recognize(final float[] pixels);
}