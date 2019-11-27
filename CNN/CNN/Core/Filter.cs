using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NN.Core;


namespace CNN.Core
{
    public class Filter : Matrix<Node<double>>, IReader, IWriter
    {
        private int? id;

        private int size, stride;
        protected fMap src;
        protected fMap trg;
        protected fMap trgIF;
        protected double[] b = new double[3];
        protected double[][] gradient;
        protected double[][] gradientStorage;
        protected string activation;
        private string formattingString = "0.000000000000";




        public Filter(int? id, int size, int stride, double stdev, double scale)
        {
            this.id = id; this.size = size; this.stride = stride;

            Configure(size, size);
            Linalg linalg = new Linalg();

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    m[i][j] = new Node<double>(linalg.RandomStandard(0, stdev, scale));
                }

            }

            gradient = new double[size][];
            for (int i = 0; i < size; i++)
                gradient[i] = new double[size];

            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    gradient[i][j] = 0;

            gradientStorage = new double[size][];
            for (int i = 0; i < size; i++)
                gradientStorage[i] = new double[size];

            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    gradientStorage[i][j] = 0;

        }

        /// <summary>
        /// aligns input and output layers
        /// </summary>
        protected virtual void Align()
        {

        }

        public static Filter Build(string type, int? id, int size, int stride, double stdev, double scale)
        {
            Filter k = null;

            switch (type)
            {
                case "conn":
                case "Connection":
                    k = new Filters.Connection(id, size, stride, stdev=1, scale);
                    break;
                case "conv":
                case "Convolution":
                    k = new Filters.Convolution(id, size, stride, stdev, scale);
                    break;
                case "img":
                case "image":
                    k = new Filters.Connection(id, size, stride, stdev=1, scale);
                    break;
                case "pool":
                case "Maxpool":
                    k = new Filters.Maxpool(id, size, stride, stdev=1, scale);
                    break;
                case "relu":
                case "Relu":
                    k = new Filters.Relu(id, size, stride, stdev, scale);
                    break;
                default:
                    throw new Exception();
            }

            return k;
        }


        public Filter forward()
        {
            if (this.GetType() == typeof(Filters.Convolution))
            {
                Linalg linalg = new Linalg();
                //Convolves filter over fmap using stride
                int s = this.stride, f = this.size;
                int curr_y = 0, out_y = 0;
                int in_dim = this.src.Size;
                while (curr_y + f <= in_dim)
                {
                    int curr_x = 0, out_x = 0;
                    while (curr_x + f <= in_dim)
                    {
                        //Create a new 5x5 matrix slice from source feature map
                        Filter slice = new Filter(null, f, s,1,1);
                        var index_y = curr_y;
                        for (int i = 0; i < f; i++)
                        {
                            var index_x = curr_x;
                            for (int j = 0; j < f; j++)
                            {
                                slice.value[i][j] = src.value[index_y][index_x];
                                index_x++;
                            }
                            index_y++;
                        }

                        Actfunc actfunc = new Actfunc();
                        Filter product = linalg.MatrixProduct(this, slice);
                        double induced_field = linalg.Sum(product) + bias[0];
                        double relu_output;
                        switch (activation)
                        {
                            case "relu":
                                relu_output = actfunc.ReLu(induced_field);
                                break;

                            default:
                                relu_output = actfunc.ReLu(induced_field);
                                break;
                        }
                        trg.InducedField[out_y][out_x] = induced_field;
                        trg.value[out_y][out_x] = new Node<double>(relu_output);
                        curr_x += s;
                        out_x += 1;
                    }
                    curr_y += s;
                    out_y += 1;
                }
            }

            if (this.GetType() == typeof(Filters.Maxpool))
            {
                int s = this.stride, f = this.size;
                int curr_y = 0, out_y = 0;
                int in_dim = this.src.Size;
                Linalg linalg = new Linalg();
                while (curr_y + f <= in_dim)
                {
                    int curr_x = 0, out_x = 0;
                    while (curr_x + f <= in_dim)
                    {
                        //Create a new 5x5 matrix slice from source feature map
                        Filter slice = new Filter(null, f, s, 1, 1);
                        var index_y = curr_y;
                        for (int i = 0; i < f; i++)
                        {
                            var index_x = curr_x;
                            for (int j = 0; j < f; j++)
                            {
                                slice.value[i][j] = src.value[index_y][index_x];
                                index_x++;
                            }
                            index_y++;
                        }

                        //Find max in slice
                        var max = linalg.Max(slice);
                        //write max value 
                        trg.value[out_y][out_x] = max;
                        curr_x += s;
                        out_x += 1;
                    }
                    curr_y += s;
                    out_y += 1;
                }

            }
            return this;
        }

        public Filter Backward(fMap fmap_previous, string activatn)
        {
            Linalg linalg = new Linalg();
            Actfunc actfunc = new Actfunc();
            int s = this.stride, f = this.size;
            double[][] filter_product = linalg.DoubleConfigure(f, f);
            double[][] fmap_product = linalg.DoubleConfigure(f, f);
            double[][] sum = linalg.DoubleConfigure(f, f);

            int curr_cy = 0, out_cy = 0;
            int pin_dim = trg.Size;
            //current layer's gradient
            while (curr_cy + f <= pin_dim)
            {
                int curr_cx = 0, out_cx = 0;
                while (curr_cx + f <= pin_dim)
                {
                    //gradient of conv(fmap)
                    fmap_product = linalg.ScalarProduct(fmap_previous.Gradient[out_cy][out_cx], this);
                    var index_cy = curr_cy;
                    for (int i = 0; i < f; i++)
                    {
                        var index_cx = curr_cx;
                        for (int j = 0; j < f; j++)
                        {
                            switch (activatn)
                            {
                                case "relu":
                                    this.trg.Gradient[index_cy][index_cx] = actfunc.DReLuConn(this.trg.InducedField[index_cy][index_cx]) * fmap_product[i][j];
                                    break;

                                default:
                                    this.trg.Gradient[index_cy][index_cx] = actfunc.DReLuConn(this.trg.InducedField[index_cy][index_cx]) * fmap_product[i][j];
                                    break;
                            }
                            index_cx++;
                        }
                        index_cy++;
                    }
                    curr_cx += s;
                    out_cx += 1;
                }
                curr_cy += s;
                out_cy += 1;
            }

            //Gradient of Filter.
            int curr_y = 0, out_y = 0;
            int in_dim = src.Size;
            while (curr_y + f <= in_dim)
            {
                int curr_x = 0, out_x = 0;
                while (curr_x + f <= in_dim)
                {
                    //Create a new 5x5 matrix slice from source feature map
                    Filter slice = new Filter(null, f, s, 1,1);
                    var index_y = curr_y;
                    for (int i = 0; i < f; i++)
                    {
                        var index_x = curr_x;
                        for (int j = 0; j < f; j++)
                        {
                            slice.value[i][j] = this.src.value[index_y][index_x];
                            index_x++;
                        }
                        index_y++;
                    }
                    //filter
                    filter_product = linalg.ScalarProduct(trg.Gradient[out_y][out_x], slice);
                    sum = linalg.AddMatrices(sum, filter_product);
                    curr_x += s;
                    out_x += 1;
                }
                curr_y += s;
                out_y += 1;
            }
            //final gradient of filter
            for (int i = 0; i < f; i++)
            {
                for (int j = 0; j < f; j++)
                {
                    this.Gradient[i][j] = sum[i][j];
                    this.GradientStorage[i][j] += sum[i][j];
                }
            }
            //gradient of bias
            bias[1] = linalg.DoubleSum(trg.Gradient);
            bias[2] += linalg.DoubleSum(trg.Gradient);
            return this;
        }

        public Filter Maxbackward(fMap fmap_previous, int s)
        {
            Linalg linalg = new Linalg();
            Actfunc actfunc = new Actfunc();
            int f = this.size;
            double[][] filter_product = linalg.DoubleConfigure(f, f);
            double[][] sum = linalg.DoubleConfigure(f, f);
            int[] indices = new int[2];

            int curr_cy = 0, out_cy = 0;
            int pin_dim = trg.Size;
            while (curr_cy + f <= pin_dim)
            {
                int curr_cx = 0, out_cx = 0;
                while (curr_cx + f <= pin_dim)
                {
                    Filter slice = new Filter(null, f, s, 1, 1);

                    var index_cy = curr_cy;
                    for (int i = 0; i < f; i++)
                    {
                        var index_cx = curr_cx;
                        for (int j = 0; j < f; j++)
                        {
                            slice.value[i][j] = trg.value[index_cy][index_cx];
                            index_cx++;
                        }
                        index_cy++;
                    }
                    indices = linalg.NanArgMax(slice);
                    trg.Gradient[curr_cy + indices[0]][curr_cx + indices[1]] = fmap_previous.Gradient[out_cy][out_cx];
                    curr_cx += s;
                    out_cx += 1;
                }
                curr_cy += s;
                out_cy += 1;
            }


            //Gradient of Filter.
            int curr_y = 0, out_y = 0;
            int in_dim = src.Size;
            while (curr_y + f <= in_dim)
            {
                int curr_x = 0, out_x = 0;
                while (curr_x + f <= in_dim)
                {
                    //Create a new 5x5 matrix slice from source feature map
                    Filter slice = new Filter(null, f, s, 1, 1);
                    var index_y = curr_y;
                    for (int i = 0; i < f; i++)
                    {
                        var index_x = curr_x;
                        for (int j = 0; j < f; j++)
                        {
                            slice.value[i][j] = this.src.value[index_y][index_x];
                            index_x++;
                        }
                        index_y++;
                    }
                    //filter
                    filter_product = linalg.ScalarProduct(trg.Gradient[out_y][out_x], slice);
                    sum = linalg.AddMatrices(sum, filter_product);
                    curr_x += s;
                    out_x += 1;
                }
                curr_y += s;
                out_y += 1;
            }
            //final gradient of filter
            for (int i = 0; i < f; i++)
            {
                for (int j = 0; j < f; j++)
                {
                    this.Gradient[i][j] = sum[i][j];
                    this.GradientStorage[i][j] += sum[i][j];
                }
            }
            //gradient of bias
            bias[1] = linalg.DoubleSum(trg.Gradient);
            //store gradient
            bias[2] += linalg.DoubleSum(trg.Gradient);
            return this;
        }

        public int? ID
        {
            get { return id; }
        }

        public fMap Source
        {
            get { return src; }

            set
            {
                src = value;
            }
        }

        public fMap Target
        {
            get { return trg; }

            set
            {
                trg = value;
            }
        }

        public fMap TargetIF
        {
            get { return trgIF; }

            set
            {
                trgIF = value;
            }
        }

        public double[][] Gradient
        {
            get { return gradient; }

            set
            {
                gradient = value;
            }
        }

        public double[][] GradientStorage
        {
            get { return gradientStorage; }

            set
            {
                gradientStorage = value;
            }
        }

        public double GradientStorageReset
        {
            set
            {
                for (int i = 0; i < size; i++)
                    for (int j = 0; j < size; j++)
                        gradientStorage[i][j] = value;
            }
        }

        public override string ToString()
        {
            string s = "";

            for (int i = 0; i < size; i++)
            {
                s += "\n";

                for (int j = 0; j < size; j++)
                    s += m[i][j].Value.ToString(formattingString) + " ";
            }

            return s;
        }

        public string ToGString()
        {
            string s = "";

            for (int i = 0; i < size; i++)
            {
                s += "\n";

                for (int j = 0; j < size; j++)
                    s += gradient[i][j].ToString(formattingString) + " ";
            }

            return s;
        }

        public string ToGradientStoreString()
        {
            string s = "";

            for (int i = 0; i < size; i++)
            {
                s += "\n";

                for (int j = 0; j < size; j++)
                    s += gradientStorage[i][j].ToString(formattingString) + " ";
            }

            return s;
        }
        public double[] bias
        {
            get { return b; }

            set
            {
                b = value;
            }
        }

        public string Activation
        {
            get { return activation; }

            set
            {
                activation = value;
            }
        }

        public Node<double>[][] value
        {
            get { return m.ToArray(); }

        }


        public int Size
        {
            get { return size; }
        }

        public int Stride
        {
            get { return stride; }
        }
    }
}