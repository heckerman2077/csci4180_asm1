import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class NgramInitialCount {

  public static class TokenizerMapper
      extends Mapper<Object, Text, Text, MapWritable> {

    private Text word_initial = new Text();
    private MapWritable map_writable = new MapWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      // get argument N from configuration
      // will not manually check if N is parsable as an int
      // or if it is a positive integer
      Configuration conf = context.getConfiguration();
      int N = Integer.parseInt(conf.get("N"));

      // define StringTokenizer custom delimiter
      // non-alphabet ascii characters
      String delim = " \t\n\r\f1234567890!\"#$&'()+,./:;<=?@[\\]^_`{|}~";
      StringTokenizer itr = new StringTokenizer(value.toString(), delim);

      // get the first ngram
      char initial_ngram[] = new char[N];
      System.out.println("[start of hdfs block]");
      for (int i = 0; i < N && itr.hasMoreTokens(); i++) {
        String token = itr.nextToken();
        System.out.printf("during init: %s\n", token);
        initial_ngram[i] = token.charAt(0);
      }

      HashMap<Character, HashMap<String, Integer>> m = new HashMap<>();

      // initialize Ngram class
      Ngram ngram = new Ngram(initial_ngram);

      // first ngram
      ngram.storeToMap(m);

      while (itr.hasMoreTokens()) {
        String token = itr.nextToken();
        System.out.println(token);
        char ch = token.charAt(0);
        // update the ngram
        // char ch = itr.nextToken().charAt(0);
        ngram.InsertAndShift(ch);

        ngram.storeToMap(m);
      }
      for (Map.Entry<Character, HashMap<String, Integer>> e : m.entrySet()) {
        word_initial.set(e.getKey().toString());
        for (Map.Entry<String, Integer> f : e.getValue().entrySet()) {
          map_writable.put(new Text(f.getKey()), new IntWritable(f.getValue()));
        }
        context.write(word_initial, map_writable);
        map_writable.clear();
      }
    }

    private static class Ngram {
      int n;
      private char data[];
      private int head;

      // data will reference the same underlying array from initial_data
      public Ngram(char[] initial_data) {
        this.data = initial_data;
        this.n = this.data.length;
        this.head = 0;
      }

      public String getAsString() {
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0, index = head; i < n; i++, index = (index + 1) % n) {
          sb.append(data[index]);
        }
        return sb.toString();
      }

      public void InsertAndShift(char ch) {
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
          if (existing_map.containsKey(rest_of_initials)) {
            existing_map.put(rest_of_initials, existing_map.get(rest_of_initials) + 1);
          } else {
            existing_map.put(rest_of_initials, 1);
          }
        } else {
          HashMap<String, Integer> new_map = new HashMap<>();
          new_map.put(rest_of_initials, 1);
          m.put(first_initial, new_map);
        }
      }
    }
  }

  public static class StripeReducer
      extends Reducer<Text, MapWritable, Text, IntWritable> {
    private Text initials_writable = new Text();
    private IntWritable count_writable = new IntWritable();

    public void reduce(Text key, Iterable<MapWritable> values,
        Context context) throws IOException, InterruptedException {
      String first_initial = key.toString();
      HashMap<String, Integer> m = new HashMap<>();

      for (MapWritable map_writable : values) {
        for (Map.Entry<Writable, Writable> e : map_writable.entrySet()) {
          String rest_of_initials = ((Text) e.getKey()).toString();
          int count = ((IntWritable) e.getValue()).get();

          String initials_tuple = first_initial.concat(rest_of_initials);
          if (m.containsKey(initials_tuple)) {
            m.put(initials_tuple, m.get(initials_tuple) + count);
          } else {
            m.put(initials_tuple, count);
          }
        }
      }

      for (Map.Entry<String, Integer> e : m.entrySet()) {
        initials_writable.set(e.getKey().replace("", " ").trim());
        count_writable.set(e.getValue());
        context.write(initials_writable, count_writable);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    // will throw exception if args[2] does not exist or is not a positive integer
    conf.set("N", args[2]);
    Job job = Job.getInstance(conf, "ngram initials count");
    job.setJarByClass(NgramInitialCount.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(MapWritable.class);
    // job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(StripeReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
