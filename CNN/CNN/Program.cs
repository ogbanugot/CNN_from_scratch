using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNN.Core;
using NN.Core;
using Drawing = System.Drawing;

namespace CNN
{
    public static class MainClass
    {
        public static void Main(string[] args)
        {

            //MainClass mainClass = new MainClass();
            //TestMatrixProduct();
            //Drawing.Bitmap SourceImage = new Drawing.Bitmap("/home/ugot/mnist/test/Penguins.jpg", true);
            //ColourtoBW(SourceImage);
            //ReadMnist();
            //mainClass.max_test();
            //model_test2();
            model_test();
            //mainClass.CCE_loss_test();
        }

        public static void max_test()
        {
            Linalg linalg = new Linalg();
            Filter m = new Filter(null, 4, 2,1,1);
            m.value[0][0] = new Node<double>(3.0); m.value[0][1] = new Node<double>(7.0); m.value[0][2] = new Node<double>(2.0); m.value[0][3] = new Node<double>(5.0);
            m.value[1][0] = new Node<double>(1.0); m.value[1][1] = new Node<double>(8.0); m.value[1][2] = new Node<double>(4.0); m.value[1][3] = new Node<double>(2.0);
            m.value[2][0] = new Node<double>(2.0); m.value[2][1] = new Node<double>(1.0); m.value[2][2] = new Node<double>(9.0); m.value[2][3] = new Node<double>(3.0);
            m.value[3][0] = new Node<double>(15.0); m.value[3][1] = new Node<double>(4.0); m.value[3][2] = new Node<double>(7.0); m.value[3][3] = new Node<double>(1.0);
            var max = linalg.Max(m);
            Console.WriteLine(max.Value);

        }

        public static CNN.Core.Model model_test()
        {
            double lr = 0.01;
            double beta1 = 0.95;
            double beta2 = 0.99;
            int batch_size = 32;
            int num_epochs = 2;
            List<List<double[][]>> dataset = ReadMnistTrain(33,78);
            //dataset = NormalizeMNIST(dataset, 33, 78);

            string configStr = "";
            // configStr += "image(size=256);";
            configStr += "conv(size=5, depth=8, stride=1, fmapsize=24, bias=0, activation=relu); ";
            configStr += "conv(size=5, depth=8, stride=1, fmapsize=20, bias=0, activation=relu); ";
            configStr += "pool(size=2, depth=8, stride=2, fmapsize=10, bias=0, activation=None); ";
            configStr += "conn(size=1, depth=128, stride=2, fmapsize=1, bias=0, activation=relu);";
            configStr += "conn(size=1, depth=10, stride=2, fmapsize=1, bias=0, activation=softmax);";
            Image image = new Image().Configure(28, 28);
            CNN.Core.Model model = new CNN.Core.Model().Configure(configStr, image);
            model.Train(dataset, lr, beta1, beta2, batch_size, num_epochs);
            return model;
        }

        public static CNN.Core.Model model_test2()
        {
            string configStr = "";
            // configStr += "image(size=256);";
            configStr += "conv(size=2, depth=1, stride=1, fmapsize=5, bias=0, activation=relu); ";
            configStr += "conv(size=2, depth=1, stride=1, fmapsize=4, bias=0, activation=relu); ";
            configStr += "pool(size=2, depth=1, stride=2, fmapsize=3, bias=0, activation=None); ";
            configStr += "conn(size=1, depth=1, stride=2, fmapsize=1, bias=0, activation=relu);";
            configStr += "conn(size=1, depth=1, stride=2, fmapsize=1, bias=0, activation=softmax);";
            Image image = new Image().Configure(6, 6);
            CNN.Core.Model model = new CNN.Core.Model().Configure(configStr, image);
            double[][] label = new double[0][];
            label[0][0] = 0.0;
            label[0][1] = 0.0;
            label[0][2] = 0.0;
            label[0][3] = 1.0;
            label[0][4] = 0.0;
            label[0][5] = 0.0;
            label[0][6] = 0.0;
            label[0][7] = 0.0;
            label[0][8] = 0.0;
            label[0][9] = 0.0;
            //Get one MNIST image
            double[][] pixels = OneMNIST();
            model.Forward(pixels);
            //calculate loss
            model.Backward(label);
            return model;
        }
        public static void random_gaussian_test()
        {
            Random rand = new Random(); //reuse this if you are generating many
            double mean = 0;
            double stdDev = 0.047855339;
            double u1 = rand.NextDouble(); //uniform(0,1] random doubles
            Console.WriteLine(u1);
            double u2 = rand.NextDouble();
            Console.WriteLine(u2);
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + stdDev * randStdNormal;
            //Console.WriteLine(randStdNormal);
            //Console.WriteLine(randNormal);
        }

        public static void TestMatrixProduct()
        {
            Linalg linalg = new Linalg();
            Filter filt1 = new Filter(null,2, 1, 1, 1);
            Filter filt2 = new Filter(null, 2, 1, 1, 1);
            filt1.value[0][0].Value = 2;
            filt1.value[0][1].Value = 3;
            filt1.value[1][0].Value = 1;
            filt1.value[1][1].Value = 4;

            filt2.value[0][0].Value = 1;
            filt2.value[0][1].Value = 4;
            filt2.value[1][0].Value = 2;
            filt2.value[1][1].Value = 3;

            Filter product = linalg.MatrixProduct(filt1, filt2);

            string s = product.ToString();

            Console.WriteLine(s);
        }

        public static List<double> Softmax_test(List<double> output)
        {
            List<double> vs = new List<double>();
            List<double> softmax = new List<double>();

            for (int i = 0; i < output.Count; i++)
            {
                vs.Add(Math.Exp(output[i]));
            }
            var total = vs.Sum();
            for (int i = 0; i < vs.Count; i++)
            {
                softmax.Add(vs[i] / total);
            }
            return softmax;

            //List<double> X = new List<double>();
            //X.Add(-0.00014549);
            //X.Add(-0.00120929);
            //X.Add(0.00356441);
            //X.Add(0.00181518);
            //X.Add(0.00325538);
            //X.Add(-0.00260445);
            //X.Add(0.00170222);
            //X.Add(0.0007031);
            //X.Add(-0.00151826);
            //X.Add(-0.00176412);
            //var r = mainClass.Softmax_test(X);
            //Console.WriteLine(r[1]);
        }

        public static double CategoricalCrossEntropy(List<double> probs, List<double> label)
        {
            if (probs.Count != label.Count)
                throw new Exception("Size mismatch: Label and probs should be the same ");
            double sum = 0;

            for (int i = 0; i < probs.Count; i++)
            {
                sum += label[i] * Math.Log(probs[i]);
            }
            return -sum;
        }

        public static void CCE_loss_test()
        {
            List<double> X = new List<double>();
            X.Add(0.10007752);
            X.Add(0.09994051);
            X.Add(0.10029686);
            X.Add(0.09974462);
            X.Add(0.09997419);
            X.Add(0.09996691);
            X.Add(0.09984748);
            X.Add(0.1000021);
            X.Add(0.1001067);
            X.Add(0.1000431);

            List<double> label = new List<double>();
            label.Add(1.0);
            label.Add(0.0);
            label.Add(0.0);
            label.Add(0.0);
            label.Add(0.0);
            label.Add(0.0);
            label.Add(0.0);
            label.Add(0.0);
            label.Add(0.0);
            label.Add(0.0);

            Console.WriteLine(CategoricalCrossEntropy(X, label));
        }

        public static double[][] OneMNIST()
        {
            FileStream ifsImages = new FileStream("/home/ugot/mnist/t10k-images.idx3-ubyte", FileMode.Open); // test images
            BinaryReader brImages = new BinaryReader(ifsImages);

            double[][] pixels = new double[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new double[28];

            // image
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    byte b = brImages.ReadByte();
                    pixels[i][j] = (double)b;
                }
            }

            return pixels;
        }

        public static void ReadMnist()
        {
            try
            {
                Console.WriteLine("\nBegin\n");
                FileStream ifsLabels = new FileStream("/home/ugot/mnist/train-labels.idx1-ubyte", FileMode.Open); // train labels
                FileStream ifsImages = new FileStream("/home/ugot/mnist/train-images.idx3-ubyte", FileMode.Open); // train images
                BinaryReader brLabels = new BinaryReader(ifsLabels);
                BinaryReader brImages = new BinaryReader(ifsImages);
                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();
                
                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();


                byte[][] pixels = new byte[28][];
                for (int i = 0; i < pixels.Length; ++i)
                    pixels[i] = new byte[28];

                // each test image
                for (int di = 0; di < 10000; ++di)
                {
                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i][j] = b;
                        }
                    }
                    byte lbl = brLabels.ReadByte();
                    DigitImage dImage = new DigitImage(pixels, lbl);
                    Console.WriteLine(dImage.ToString());
                    Console.ReadLine();
                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();

                Console.WriteLine("\nEnd\n");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }
        }

        public static List<List<double[][]>> ReadMnistTrain(double mean, double std)
        {
            try
            {
                FileStream ifsLabels = new FileStream("/home/ugot/mnist/train-labels.idx1-ubyte", FileMode.Open); // train labels
                FileStream ifsImages = new FileStream("/home/ugot/mnist/train-images.idx3-ubyte", FileMode.Open); // train images
                BinaryReader brLabels = new BinaryReader(ifsLabels);
                BinaryReader brImages = new BinaryReader(ifsImages);
                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();
                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                List<double[][]> label = new List<double[][]>();
                List<double[][]> image = new List<double[][]>();
                List<List<double[][]>> dataset = new List<List<double[][]>>();
                // each test image
                for (int di = 0; di < 50000; ++di)
                {
                    double[][] pixels = new double[28][];
                    for (int i = 0; i < 28; ++i)
                        pixels[i] = new double[28];

                    for (int i = 0; i < 28; ++i)
                    {
                        for (int j = 0; j < 28; ++j)
                        {
                            //Normalize
                            double b = brImages.ReadByte();
                            double c = b - mean;
                            double d = c / std;
                            pixels[i][j] = d;
                        }
                    }

                    byte lbl = brLabels.ReadByte();
                    double[][] lbl_encoded = LabelEncode(lbl);
                    label.Add(lbl_encoded);
                    image.Add(pixels);
                } // each image
                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();
                dataset.Add(image);
                dataset.Add(label);
                return dataset;

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
                throw new Exception();
            }

        }

        public static List<List<double[][]>> NormalizeMNIST(List<List<double[][]>> dataset, double mean, double std)
        {
            //Substract mean from dataset values and Divide dataset values by standard deviation
            for (int i = 0; i < dataset[0].Count; i++)
            {
                for (int j = 0; j < dataset[0][i].Length; j++)
                {
                    for (int l = 0; l < dataset[0][i][0].Length; l++)
                    {
                        dataset[0][i][j][l] = dataset[0][i][j][l] - mean;
                        dataset[0][i][j][l] = dataset[0][i][j][l]/std;

                    }
                }
            }

            return dataset;
        }

        public static double[][] LabelEncode(int lbl)
        {
            double[][] zero = { new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

            double[][] one = { new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 } };
            double[][] two = { new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 } };
            double[][] three = { new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 } };
            double[][] four = { new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 } };
            double[][] five = { new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 } };
            double[][] six = { new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 } };
            double[][] seven = { new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 } };
            double[][] eight = { new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 } };
            double[][] nine = { new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } };

            switch (lbl)
            {
                case 0:
                    return zero;
                case 1:
                    return one;
                case 2:
                    return two;
                case 3:
                    return three;
                case 4:
                    return four;
                case 5:
                    return five;
                case 6:
                    return six;
                case 7:
                    return seven;
                case 8:
                    return eight;
                case 9:
                    return nine;
                default:
                    throw new Exception("Number must be between 0 and 9");
            }
        }


        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ColourtoBW(Drawing.Bitmap SourceImage)
        {
            Drawing.Graphics gr = Drawing.Graphics.FromImage(SourceImage); // SourceImage is a Bitmap object
            var threshold = 0.8f;
            var gray_matrix = new float[][] {
                        new float[] { 0.299f, 0.299f, 0.299f, 0, 0 },
                        new float[] { 0.587f, 0.587f, 0.587f, 0, 0 },
                        new float[] { 0.114f, 0.114f, 0.114f, 0, 0 },
                        new float[] { 0,      0,      0,      1, 0 },
                        new float[] { 0,      0,      0,      0, 1 }
                    };

            var ia = new Drawing.Imaging.ImageAttributes();
            ia.SetColorMatrix(new Drawing.Imaging.ColorMatrix(gray_matrix));
            ia.SetThreshold(threshold); // Change this threshold as needed
            var rc = new Drawing.Rectangle(0, 0, SourceImage.Width, SourceImage.Height);
            gr.DrawImage(SourceImage, rc, 0, 0, SourceImage.Width, SourceImage.Height, Drawing.GraphicsUnit.Pixel, ia);
            ConsoleWriteImage(SourceImage);
            Console.ReadLine();
        }

        static int[] cColors = { 0x000000, 0x000080, 0x008000, 0x008080, 0x800000, 0x800080, 0x808000, 0xC0C0C0, 0x808080, 0x0000FF, 0x00FF00, 0x00FFFF, 0xFF0000, 0xFF00FF, 0xFFFF00, 0xFFFFFF };

        public static void ConsoleWritePixel(Drawing.Color cValue)
        {
            Drawing.Color[] cTable = cColors.Select(x => Drawing.Color.FromArgb(x)).ToArray();
            char[] rList = new char[] { (char)9617, (char)9618, (char)9619, (char)9608 }; // 1/4, 2/4, 3/4, 4/4
            int[] bestHit = new int[] { 0, 0, 4, int.MaxValue }; //ForeColor, BackColor, Symbol, Score

            for (int rChar = rList.Length; rChar > 0; rChar--)
            {
                for (int cFore = 0; cFore < cTable.Length; cFore++)
                {
                    for (int cBack = 0; cBack < cTable.Length; cBack++)
                    {
                        int R = (cTable[cFore].R * rChar + cTable[cBack].R * (rList.Length - rChar)) / rList.Length;
                        int G = (cTable[cFore].G * rChar + cTable[cBack].G * (rList.Length - rChar)) / rList.Length;
                        int B = (cTable[cFore].B * rChar + cTable[cBack].B * (rList.Length - rChar)) / rList.Length;
                        int iScore = (cValue.R - R) * (cValue.R - R) + (cValue.G - G) * (cValue.G - G) + (cValue.B - B) * (cValue.B - B);
                        if (!(rChar > 1 && rChar < 4 && iScore > 50000)) // rule out too weird combinations
                        {
                            if (iScore < bestHit[3])
                            {
                                bestHit[3] = iScore; //Score
                                bestHit[0] = cFore;  //ForeColor
                                bestHit[1] = cBack;  //BackColor
                                bestHit[2] = rChar;  //Symbol
                            }
                        }
                    }
                }
            }
            Console.ForegroundColor = (ConsoleColor)bestHit[0];
            Console.BackgroundColor = (ConsoleColor)bestHit[1];
            Console.Write(rList[bestHit[2] - 1]);
        }


        public static void ConsoleWriteImage(Drawing.Bitmap source)
        {
            int sMax = 39;
            decimal percent = Math.Min(decimal.Divide(sMax, source.Width), decimal.Divide(sMax, source.Height));
            Drawing.Size dSize = new Drawing.Size((int)(source.Width * percent), (int)(source.Height * percent));
            Drawing.Bitmap bmpMax = new Drawing.Bitmap(source, dSize.Width * 2, dSize.Height);
            for (int i = 0; i < dSize.Height; i++)
            {
                for (int j = 0; j < dSize.Width; j++)
                {
                    ConsoleWritePixel(bmpMax.GetPixel(j * 2, i));
                    ConsoleWritePixel(bmpMax.GetPixel(j * 2 + 1, i));
                }
                System.Console.WriteLine();
            }
            Console.ResetColor();
        }

    }

    public class DigitImage
    {
        public byte[][] pixels;
        public byte label;

        public DigitImage(byte[][] pixels,
          byte label)
        {
            this.pixels = new byte[28][];
            for (int i = 0; i < this.pixels.Length; ++i)
                this.pixels[i] = new byte[28];

            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    this.pixels[i][j] = pixels[i][j];

            this.label = label;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (this.pixels[i][j] == 0)
                        s += " "; // white
                    else if (this.pixels[i][j] == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += this.label.ToString();
            return s;
        } // ToString

    }

}
