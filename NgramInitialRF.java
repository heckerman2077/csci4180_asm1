import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class NgramInitialRF {

  public static class TokenizerMapper
      extends Mapper<Object, Text, Text, MapWritable> {

    private Text word_initial = new Text();
    private MapWritable map_writable = new MapWritable();

    private Ngram ngram;
    private HashMap<Character, HashMap<String, Integer>> m = new HashMap<>();

    public void setup(Context context) throws IOException, InterruptedException {
      // get argument N from configuration
      // will not manually check if N is parsable as an int
      // or if it is a positive integer
      Configuration conf = context.getConfiguration();
      int N = Integer.parseInt(conf.get("N"));

      // initialze ngram class
      ngram = new Ngram(N);
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
      // emit the combined values once
      for (Map.Entry<Character, HashMap<String, Integer>> e : m.entrySet()) {
        word_initial.set(e.getKey().toString());
        for (Map.Entry<String, Integer> f : e.getValue().entrySet()) {
          map_writable.put(
              new Text(f.getKey()),
              new IntWritable(f.getValue()));
        }
        context.write(word_initial, map_writable);
        map_writable.clear();
      }
    }

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      // define StringTokenizer custom delimiter
      // non-alphabet ascii characters
      String delim = " \t\n\r\f!\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~";
      StringTokenizer itr = new StringTokenizer(value.toString(), delim);

      // initialize ngram with (n - 1) elements if not initialized
      while (ngram.notInitialized() && itr.hasMoreTokens()) {
        // get the first character of the next token
        char ch = itr.nextToken().charAt(0);
        ngram.initialize(ch);
      }

      while (itr.hasMoreTokens()) {
        // get the first character of the next token
        char ch = itr.nextToken().charAt(0);

        // update the ngram
        ngram.InsertAndShift(ch);

        // store ngram to hashmap
        ngram.storeToMap(m);
      }
    }

    private static class Ngram {
      int n;
      private char data[];
      private int initialize_count;
      private int head;

      // data will reference the same underlying array from initial_data
      private Ngram(int n) {
        this.data = new char[n];
        this.initialize_count = 0;
        this.n = n;
        this.head = 0;
      }

      // notInitialized() will return true when ngram has less than (n - 1) elements
      private boolean notInitialized() {
        return initialize_count < n - 1;
      }

      private void initialize(char ch) {
        this.InsertAndShift(ch);
        initialize_count++;
      }

      private String getAsString() {
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0, index = head; i < n; i++, index = (index + 1) % n) {
          sb.append(data[index]);
        }
        return sb.toString();
      }

      private void InsertAndShift(char ch) {
        data[head] = ch;
        head = (head + 1) % n;
      }

      // stores to map
      // {first_initial={rest_of_initials=1}}
      // or increments the Integer value if the rest_of_initials key exists
      private void storeToMap(HashMap<Character, HashMap<String, Integer>> m) {
        String ngram_str = this.getAsString();
        char first_initial = ngram_str.charAt(0);
        String rest_of_initials = ngram_str.substring(1);

        if (m.containsKey(first_initial)) {
          HashMap<String, Integer> existing_map = m.get(first_initial);
          existing_map.put(
              rest_of_initials,
              existing_map.containsKey(rest_of_initials) ? existing_map.get(rest_of_initials) + 1 : 1);
        } else {
          HashMap<String, Integer> new_map = new HashMap<>();
          new_map.put(rest_of_initials, 1);
          m.put(first_initial, new_map);
        }
      }
    }
  }

  public static class StripeReducer
      extends Reducer<Text, MapWritable, Text, FloatWritable> {
    private Text initials_writable = new Text();
    private FloatWritable relatibe_frq_writable = new FloatWritable();

    public void reduce(Text key, Iterable<MapWritable> values, Context context)
        throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      double theta = Double.parseDouble(conf.get("theta"));

      String first_initial = key.toString();
      HashMap<String, Integer> m = new HashMap<>();

      int total_count = 0;
      for (MapWritable map_writable : values) {
        for (Map.Entry<Writable, Writable> e : map_writable.entrySet()) {
          String rest_of_initials = ((Text) e.getKey()).toString();
          int count = ((IntWritable) e.getValue()).get();
          total_count += count;

          String initials_tuple = first_initial.concat(rest_of_initials);
          m.put(
              initials_tuple,
              m.containsKey(initials_tuple) ? m.get(initials_tuple) + count : count);
        }
      }

      for (Map.Entry<String, Integer> e : m.entrySet()) {
        initials_writable.set(e.getKey().replace("", " ").trim());
        int count = e.getValue();
        float relative_freq = (float) count / (float) total_count;
        if (relative_freq < theta) {
          continue;
        }
        relatibe_frq_writable.set(relative_freq);
        context.write(initials_writable, relatibe_frq_writable);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    // will throw exception is args[2] does not exist or is not a positive integer
    conf.set("N", args[2]);
    conf.set("theta", args[3]);
    Job job = Job.getInstance(conf, "ngram initials relative frequency");
    job.setJarByClass(NgramInitialRF.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(MapWritable.class);
    // job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(StripeReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(FloatWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
