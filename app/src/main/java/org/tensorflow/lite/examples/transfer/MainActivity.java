package org.tensorflow.lite.examples.transfer;

import android.Manifest;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import pub.devrel.easypermissions.EasyPermissions;

enum Mode {
    Data_Collection,
    Inference,
    Training

}

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    // for the base model
    int SMPSready = 0;
    private List<Classifier> mClassifiers = new ArrayList<>();

    /*KNN vars begin*/
    String KNNActivity = "Nothing";
    String KNNFILE_NAME= "KNN_rec";
    String HeadFILE_NAME= "Head_rec";
    String BaseFILE_NAME= "Base_rec";
    List<String> ActivityTypes = Arrays.asList("downstairs", "jogging", "running", "standing", "upstairs", "walking");

    int sampleCount = 0, windowLength = 50, Fs = 40;
    int FEATURE_LENGTH = 12, SAMPLE_LENGTH = 118;

    float[] acc_X_array = new float[windowLength];
    float[] acc_Y_array = new float[windowLength];
    float[] acc_Z_array = new float[windowLength];
    float accelerometer_X_value = 0, accelerometer_Y_value = 0, accelerometer_Z_value = 0;
    String[] categoryArray = new String[SAMPLE_LENGTH];
    double[][] trainedSampleTable = new double[SAMPLE_LENGTH][FEATURE_LENGTH];
    FileOutputStream KNNModelfos = null;
    FileOutputStream HeadModelfos = null;
    FileOutputStream BaseModelfos = null;
    private static final byte isAnalysing = 2;
    private static final byte isIdle = -1;
    byte systemState = isIdle;
    // functions class init
    functions Functions = new functions();
    /*KNN vars end*/

    List<String> HeadActivityTypes = Arrays.asList("Upstairs", "Downstairs", "Walking", "Jogging", "Sitting", "Standing");
    float[] confidenceList = new float[ActivityTypes.size()];
    final int NUM_SAMPLES = 80;
    double VB_THRESHOLD = 0.75;

    //'Downstairs','Jogging','Sitting','Standing','Upstairs','Walking'
    int DownstairsInstanceCount = 0;
    int JoggingInstanceCount = 0;
    int SittingInstanceCount = 0;
    int StandingInstanceCount = 0;
    int UpstairsInstanceCount = 0;
    int WalkingInstanceCount = 0;

    boolean isRunning = false;
    String finalVoteHeadstring = "Nothing";
    SensorManager mSensorManager;
    Sensor mAccelerometer;
    TransferLearningModelWrapper tlModel;
    String classId;

    static List<Float> x_accel;
    static List<Float> y_accel;
    static List<Float> z_accel;
    static List<Float> x_accel_normed = new ArrayList<>();
    static List<Float> y_accel_normed = new ArrayList<>();
    static List<Float> z_accel_normed = new ArrayList<>();

    static List<Float> input_signal;

    Mode mode;

    Button startButton;
    Button stopButton;
    TextView lossValueTextView;
    Spinner optionSpinner;
    Spinner classSpinner;
    Vibrator vibrator;

    TextView KNNFinalVote;
    TextView BaseinalVote;
    TextView HeadfinalVote;
    TextView TextKNNDetectedActivity;
    TextView TextBaseDetectedActivity;
    TextView TextHeadDetectedActivity;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        startButton = (Button) findViewById(R.id.buttonStart);
        stopButton = (Button) findViewById(R.id.buttonStop);
        stopButton.setEnabled(false);

        KNNFinalVote = (TextView) findViewById(R.id.knnFinalVote);
        BaseinalVote = (TextView) findViewById(R.id.baseFinalVote);
        HeadfinalVote = (TextView) findViewById(R.id.headFinalVote);

        TextKNNDetectedActivity = (TextView) findViewById(R.id.label_Activity_KNN);
        TextBaseDetectedActivity = (TextView) findViewById(R.id.label_Activity_Base);
        TextHeadDetectedActivity = (TextView) findViewById(R.id.label_Activity_Head);

        TextKNNDetectedActivity.setMovementMethod(new ScrollingMovementMethod());
        TextBaseDetectedActivity.setMovementMethod(new ScrollingMovementMethod());
        TextHeadDetectedActivity.setMovementMethod(new ScrollingMovementMethod());


        lossValueTextView = (TextView) findViewById(R.id.lossValueTextView);

        optionSpinner = (Spinner) findViewById(R.id.optionSpinner);
        classSpinner = (Spinner) findViewById(R.id.classSpinner);

        ArrayAdapter<CharSequence> optionAdapter = ArrayAdapter
                .createFromResource(this, R.array.options_array,
                        R.layout.spinner_item);
        optionAdapter
                .setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        optionSpinner.setAdapter(optionAdapter);

        ArrayAdapter<CharSequence> classAdapter = ArrayAdapter
                .createFromResource(this, R.array.class_array,
                        R.layout.spinner_item);
        classAdapter
                .setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        classSpinner.setAdapter(classAdapter);

        x_accel = new ArrayList<Float>();
        y_accel = new ArrayList<Float>();
        z_accel = new ArrayList<Float>();
        input_signal = new ArrayList<Float>();

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        //mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        tlModel = new TransferLearningModelWrapper(getApplicationContext());

        optionSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view,
                                       int position, long id) {
                String option = (String) parent.getItemAtPosition(position);
                switch (option) {
                    case "Data Collection":
                        mode = Mode.Data_Collection;
                        break;
                    case "Training":
                        mode = Mode.Training;
                        break;
                    case "Inference":
                        mode = Mode.Inference;
                        break;
                    default:
                        throw new IllegalArgumentException("Invalid app mode.");
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // TODO Auto-generated method stub
            }
        });

        classSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view,
                                       int position, long id) {
                classId = (String) parent.getItemAtPosition(position);

            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // TODO Auto-generated method stub
            }
        });

        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startButton.setEnabled(false);
                stopButton.setEnabled(true);
                optionSpinner.setEnabled(false);
                isRunning = true;
                // Uncomment following lines to load an existing model.
                if (mode == Mode.Inference) {
                    systemState = isAnalysing;
                    streamInit();
                    analyze(v);

                    File modelPath = getApplicationContext().getFilesDir();
                    File modelFile = new File(modelPath, "inference.tflite");
                    if (modelFile.exists()) {
                        tlModel.loadModel(modelFile);
                        Toast.makeText(getApplicationContext(), "Model loaded.", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(getApplicationContext(), "No model has been loaded", Toast.LENGTH_SHORT).show();
                    }

                }
            }
        });

        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (KNNModelfos != null) {
                    try {
                        KNNModelfos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                if (BaseModelfos != null) {
                    try {
                        BaseModelfos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                if (HeadModelfos != null) {
                    try {
                        HeadModelfos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                systemState = isIdle;
                startButton.setEnabled(true);
                stopButton.setEnabled(false);
                optionSpinner.setEnabled(true);
                isRunning = false;
                if (mode == Mode.Training) {
                    tlModel.disableTraining();
                    // Uncomment following lines to save the model.

                    File modelPath = getApplicationContext().getFilesDir();
                    File modelFile = new File(modelPath, "MODEL_NAME");
                    tlModel.saveModel(modelFile);
                    Toast.makeText(getApplicationContext(), "Model saved.", Toast.LENGTH_SHORT).show();

                }
            }
        });
        // for KNN
        readReferenceCSV();
        // for the base model
        // TensorFlow: load up our saved model to perform inference from local storage
        loadModel();
        Timer SMPS = new Timer();
        //SMPS.cancel();
        TimerTask analyzeTask = new TimerTask() {
            @Override
            public void run() {
                SMPSready = 1;
            }
        };
        SMPS.schedule(analyzeTask, 0, 50); //20HZ smps
    }

    private void loadModel() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    mClassifiers.add(
                            TensorFlowLiteClassifier.create(getAssets(), "TensorFlowLite",
                                    "base/converted_model.tflite", "base/label.txt"));
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    private void readReferenceCSV() {
        InputStream is = getResources().openRawResource(R.raw.reference);
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(is, Charset.forName("UTF-8"))
        );
        String line = "";
        int offset = 2;
        int i = 0;

        try {
            for (int sampleLine = 0; sampleLine < SAMPLE_LENGTH; sampleLine++) {
                // row offset
                while (i < offset) {
                    i++;
                    line = reader.readLine();
                }
                // Split by ','
                String[] tokens = line.split(",");
                // Read the Category
                categoryArray[sampleLine] = tokens[0];
                // Read the table data
                for (int Col = 0; Col < 12; Col++) {
                    double currentFeature = Double.parseDouble(tokens[Col + 1]);
                    trainedSampleTable[sampleLine][Col] = currentFeature;
                }
                line = reader.readLine();

            }
            Log.d("My Activity", "Just created trainign sample");
        } catch (IOException e) {
            Log.wtf("MyActivity", "Error reading file on line " + line, e);
            e.printStackTrace();
        }


    }

    void streamInit() {
        //initiates a file to record or analyze
        String[] perms = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE};
        if (EasyPermissions.hasPermissions(this, perms)) {
            Toast.makeText(this, "has permissions", Toast.LENGTH_SHORT).show();
            KNNFILE_NAME = Functions.fileNAmeGenerator();
            HeadFILE_NAME = Functions.fileNAmeGenerator();
            BaseFILE_NAME = Functions.fileNAmeGenerator();
            switch (systemState) {
                case isAnalysing:
                    KNNFILE_NAME = "KNNConfMatrix_" + KNNFILE_NAME;
                    HeadFILE_NAME = "HeadConfMatrix_" + HeadFILE_NAME;
                    BaseFILE_NAME = "BaseConfMatrix_" + BaseFILE_NAME;
                    break;
            }

            if (isExternalStorageWritable()) {
                Context context = this;
                File KNNCSVFile = new File(context.getExternalFilesDir(null), KNNFILE_NAME);
                File HeadCSVFile = new File(context.getExternalFilesDir(null), HeadFILE_NAME);
                File BaseCSVFile = new File(context.getExternalFilesDir(null), BaseFILE_NAME);
                try {
                    KNNModelfos = new FileOutputStream(KNNCSVFile);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    systemState = isIdle;
                }
                try {
                    HeadModelfos = new FileOutputStream(HeadCSVFile);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    systemState = isIdle;
                }
                try {
                    BaseModelfos = new FileOutputStream(BaseCSVFile);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    systemState = isIdle;
                }
                Toast.makeText(this, "Recording at" + getFilesDir() + "/" + KNNFILE_NAME, Toast.LENGTH_SHORT).show();
            }

        } else {
            EasyPermissions.requestPermissions(this, "I need your permission", 123, perms);
        }

    }

    private boolean isExternalStorageWritable() {
        if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
            Log.i("State", "Yes it is writable");
            return true;
        } else {
            return false;
        }
    }

    public void analyze(View v) {
        Timer analyzeTimer = new Timer();
        if (systemState == isIdle) {
            analyzeTimer.cancel();
        }
        TimerTask analyzeTask = new TimerTask() {
            @Override
            public void run() {
                acc_X_array[sampleCount] = accelerometer_X_value;
                acc_Y_array[sampleCount] = accelerometer_Y_value;
                acc_Z_array[sampleCount] = accelerometer_Z_value;
                sampleCount++;
                if (sampleCount >= windowLength) {
                    sampleCount = 0;
                    Functions.SetVars(windowLength, Fs, FEATURE_LENGTH, SAMPLE_LENGTH,
                            acc_X_array, acc_Y_array, acc_Z_array,
                            trainedSampleTable, categoryArray);
                    Calendar calendar = Calendar.getInstance();
                    SimpleDateFormat format = new SimpleDateFormat("HH_mm_ss");
                    String time = format.format(calendar.getTime());

                    //KNN output scroll
                    KNNActivity = Functions.categorize();
                    String KnnVotingArray = Functions.getVotingArrayEU().toString();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            KNNFinalVote.setText(KnnVotingArray);
                            String gotText = TextKNNDetectedActivity.getText().toString();
                            TextKNNDetectedActivity.setText(KNNActivity + "\n" + gotText);
                        }
                    });

                    try { //KNN log to file
                        Log.d("vot", Functions.getVotingArrayEU().toString());
                        KNNModelfos.write((Functions.getVotingArrayEU().toString() + "\n").getBytes());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        analyzeTimer.schedule(analyzeTask, 0, 1000 / Fs);
    }

    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
    }

    protected void onDestroy() {
        super.onDestroy();
        tlModel.close();
        tlModel = null;
        mSensorManager = null;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        switch (event.sensor.getType()) {
            case Sensor.TYPE_ACCELEROMETER:
                accelerometer_X_value = event.values[0];
                accelerometer_Y_value = event.values[1];
                accelerometer_Z_value = event.values[1];

                if (SMPSready == 1) {
                    SMPSready = 0; //slows down the DAQ
                    x_accel.add(accelerometer_X_value);
                    y_accel.add(accelerometer_Y_value);
                    z_accel.add(accelerometer_Z_value);
                }
                break;
        }

        //Check if we have desired number of samples for sensors, if yes, the process input.
        if (x_accel.size() == NUM_SAMPLES && y_accel.size() == NUM_SAMPLES && z_accel.size() == NUM_SAMPLES) {
            //normalize
            for (int i = 0; i < x_accel.size(); i++) {
                x_accel_normed.add((float) 0); //just to init
                float norm = (x_accel.get(i) - Collections.min(x_accel)) / Collections.max(x_accel);
                x_accel_normed.set(i, norm);
            }
            for (int i = 0; i < y_accel.size(); i++) {
                y_accel_normed.add((float) 0); //just to init
                float norm = (y_accel.get(i) - Collections.min(y_accel)) / Collections.max(y_accel);
                y_accel_normed.set(i, norm);
            }
            for (int i = 0; i < z_accel.size(); i++) {
                z_accel_normed.add((float) 0); //just to init
                float norm = (z_accel.get(i) - Collections.min(z_accel)) / Collections.max(z_accel);
                z_accel_normed.set(i, norm);
            }
            processInput();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void processInput() {
        int i = 0;
        while (i < NUM_SAMPLES) {
            input_signal.add(x_accel_normed.get(i));
            input_signal.add(y_accel_normed.get(i));
            input_signal.add(z_accel_normed.get(i));
            i++;
        }

        float[] input = toFloatArray(input_signal);

        if (isRunning) {
            if (mode == Mode.Training) {
                int batchSize = tlModel.getTrainBatchSize();
                if (DownstairsInstanceCount >= batchSize && JoggingInstanceCount >= batchSize && WalkingInstanceCount >= batchSize
                        && UpstairsInstanceCount >= batchSize) {//&& SittingInstanceCount >= batchSize && StandingInstanceCount >= batchSize) {
                    tlModel.enableTraining((epoch, loss) -> runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            lossValueTextView.setText(Float.toString(round(loss, 2)));
                        }
                    }));
                } else {
                    String message = batchSize + " instances per class are required for training.";
                    Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT).show();
                    stopButton.callOnClick();
                }

            } else if (mode == Mode.Data_Collection) {
                tlModel.addSample(input, classId);
                switch (classId) {
                    case "Walking":
                        WalkingInstanceCount += 1;
                        Log.d("WalkingInstanceCount", Integer.toString(WalkingInstanceCount));
                        if (WalkingInstanceCount == 5) {
                            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
                        }
                        break;
                    case "Jogging":
                        JoggingInstanceCount += 1;
                        Log.d("JoggingInstanceCount", Integer.toString(JoggingInstanceCount));
                        if (JoggingInstanceCount == 5) {
                            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
                        }

                        break;
                    case "Upstairs":
                        UpstairsInstanceCount += 1;
                        Log.d("UpstairsInstanceCount", Integer.toString(UpstairsInstanceCount));
                        if (UpstairsInstanceCount == 5) {
                            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
                        }

                        break;
                    case "Downstairs":
                        DownstairsInstanceCount += 1;
                        Log.d("DownstairsInstanceCount", Integer.toString(DownstairsInstanceCount));
                        if (DownstairsInstanceCount == 5) {
                            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
                        }

                        break;
                    case "Sitting":
                        SittingInstanceCount += 1;
                        Log.d("SittingInstanceCount", Integer.toString(SittingInstanceCount));
                        if (SittingInstanceCount == 5) {
                            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
                        }

                        break;
                    case "Standing":
                        StandingInstanceCount += 1;
                        Log.d("StandingInstanceCount", Integer.toString(StandingInstanceCount));
                        if (StandingInstanceCount == 5) {
                            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE));
                        }

                        break;
                }

            } else if (mode == Mode.Inference) {
                //for the Head model
                TransferLearningModel.Prediction[] HEADpredictions = tlModel.predict(input);

                confidenceList[0] = round(HEADpredictions[0].getConfidence(), 2);
                confidenceList[1] = round(HEADpredictions[1].getConfidence(), 2);
                confidenceList[2] = round(HEADpredictions[2].getConfidence(), 2);
                confidenceList[3] = round(HEADpredictions[3].getConfidence(), 2);
                confidenceList[4] = -1;//round(HEADpredictions[4].getConfidence(), 2);
                confidenceList[5] = -1;//round(HEADpredictions[5].getConfidence(), 2);

                List<Float> confidenceArrayList = new ArrayList<Float>();
                confidenceArrayList.add(confidenceList[0]);
                confidenceArrayList.add(confidenceList[1]);
                confidenceArrayList.add(confidenceList[2]);
                confidenceArrayList.add(confidenceList[3]);
                //confidenceArrayList.add(confidenceList[4]);
                //confidenceArrayList.add(confidenceList[5]);

                // head model scroll
                String Activity = confidenceArrayList.toString();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        HeadfinalVote.setText(Activity);
                    }
                });

                Float finalVoteHeadIndex = Collections.max(confidenceArrayList);
                finalVoteHeadstring = HeadActivityTypes.get(confidenceArrayList.indexOf(finalVoteHeadIndex));
                String gotTextHead = TextHeadDetectedActivity.getText().toString();
                TextHeadDetectedActivity.setText(finalVoteHeadstring + "\n" + gotTextHead);
                try { //Head model log to file
                    Log.d("Head VOT", confidenceArrayList.toString());
                    HeadModelfos.write((confidenceArrayList.toString() + "\n").getBytes());
                } catch (IOException e) {
                    e.printStackTrace();
                }

                //base model scroll
                for (Classifier classifier : mClassifiers) {
                    final Classification res = classifier.recognize(input);
                    String BaseActivity = res.getLabel();
                    String gotTextBase = TextBaseDetectedActivity.getText().toString();
                    TextBaseDetectedActivity.setText(BaseActivity + "\n" + gotTextBase);
                    float[] baseVotingArray = TensorFlowLiteClassifier.getBaseVotingArray();
                    baseVotingArray[0] = round(baseVotingArray[0], 2);
                    baseVotingArray[1] = round(baseVotingArray[1], 2);
                    baseVotingArray[2] = round(baseVotingArray[2], 2);
                    baseVotingArray[3] = round(baseVotingArray[3], 2);
                    baseVotingArray[4] = round(baseVotingArray[4], 2);
                    baseVotingArray[5] = round(baseVotingArray[5], 2);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            BaseinalVote.setText(Arrays.toString(baseVotingArray));

                        }
                    });
                    try { //Base model log to file
                        Log.d("Head VOT", confidenceArrayList.toString());
                        BaseModelfos.write((Arrays.toString(baseVotingArray) + "\n").getBytes());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }
        }

        // Clear all the values
        x_accel.clear();
        y_accel.clear();
        z_accel.clear();
        input_signal.clear();
    }


    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }
}
