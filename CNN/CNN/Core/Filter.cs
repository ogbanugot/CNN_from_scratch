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
        //ioset,
        protected fMap src;
        protected fMap trg;
        protected double b;
        protected string activation;



        public Filter(int? id, int size, int stride)
        {
            this.id = id; this.size = size; this.stride = stride;

            Configure(size, size);

            for (int i = 0; i < size; i++)
            {
                Linalg linalg = new Linalg();
                for (int j = 0; j < size; j++)
                    m[i][j] = new Node<double>(linalg.RandomGaussian(0, 0.047855339));
            }
        }

        /// <summary>
        /// aligns input and output layers
        /// </summary>
        protected virtual void Align()
        {

        }

        public static Filter Build(string type, int? id, int size, int stride)
        {
            Filter k = null;

            switch (type)
            {
                case "conn":
                case "Connection":
                    k = new Filters.Connection(id, size, stride);
                    break;
                case "conv":
                case "Convolution":
                    k = new Filters.Convolution(id, size, stride);
                    break;
                case "pool":
                case "Maxpool":
                    k = new Filters.Maxpool(id, size, stride);
                    break;
                case "relu":
                case "Relu":
                    k = new Filters.Relu(id, size, stride);
                    break;
                default:
                    throw new Exception();
            }

            return k;
        }

        //public void Drive()
        //{
        //    bool nextright = true, nextdown = true;

        //    ioset = 0;

        //    if (src.Count != trg.Count)
        //        throw new Exception();

        //    // 0. align
        //    Align();

        //    for (; ioset < src.Count; ioset++)
        //    {
        //        while (nextdown)
        //        {
        //            // 1. lock reset positions
        //            trg[ioset].LockResetPosition();
        //            src[ioset].LockResetPosition();

        //            while (nextright)
        //            {
        //                // 2. compute output

        //                // 3. move right
        //                nextright = trg[ioset].Move(Notifier.Writer, Direction.Right);
        //                if (src[ioset].Move(Notifier.Reader, Direction.Right) != nextright)
        //                    throw new Exception();
        //            }

        //            // Reset()
        //            nextdown = trg[ioset].Move(Notifier.Writer, Direction.Down);
        //            if (src[ioset].Move(Notifier.Reader, Direction.Down) != nextdown)
        //                throw new Exception();
        //        }
        //    }
        //}

        public Filter forward()
        {
            if (this.GetType() == typeof(Filters.Convolution))
            {
                //Convolves filter over fmap using stride
                int s = this.stride, f = this.size;
                int curr_y = 0, out_y = 0;
                int in_dim = this.src.Size;
                while(curr_y + f <= in_dim)
                {
                    int curr_x = 0, out_x = 0;
                    while(curr_x + f <= in_dim)
                    {
                        //Create a new 5x5 matrix slice from source feature map
                        Filter slice = new Filter(null,f,s);
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

                        Linalg linalg = new Linalg();
                        Actfunc actfunc = new Actfunc();
                        Filter product = linalg.MatrixProduct(this, slice);
                        double relu_output;
                        switch (activation)
                        {
                            case "relu":
                                relu_output = actfunc.ReLu(linalg.Sum(product) + bias);
                                break;
                                                      
                            default:
                                relu_output = actfunc.ReLu(linalg.Sum(product) + bias);
                                break;
                        }
                        //double relu_output = actfunc.ReLu(linalg.Sum(product) + bias);
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
                while (curr_y + f <= in_dim)
                {
                    int curr_x = 0, out_x = 0;
                    while (curr_x + f <= in_dim)
                    {
                        //Create a new 5x5 matrix slice from source feature map
                        Filter slice = new Filter(null, f, s);
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

                        Linalg linalg = new Linalg();
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
            if (this.GetType() == typeof(Filters.Connection))
            {
                Console.WriteLine("Its a connect filter");

            }
            return this;
        }

        public Filter backward()
        {
            if (this.GetType() == typeof(Filters.Convolution))
            {

            }
            if (this.GetType() == typeof(Filters.Maxpool))
            {

            }
            if (this.GetType() == typeof(Filters.Connection))
            {

            }
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

        public double bias
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