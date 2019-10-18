using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NN.Core;

namespace CNN.Core
{
    public class Layer : ILayer
    {
        protected IList<Filter> filter = new List<Filter>();
        protected IList<fMap> fmaps = new List<fMap>();
        protected IList<IList<Filter>> connection = new List<IList<Filter>>();



        public Layer() { }

        public Layer Configure(string config, ILayer input)
        {
            string[] cfg = config.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            string[] tkn = null;

            string cfgstr = "";
            for (int i = 0; i < cfg.Length; i++)
                cfgstr += cfg[i];

            cfg = cfgstr.Split(new char[] { '[', '(', ',', ')', ']' }, StringSplitOptions.RemoveEmptyEntries);

            // syntax: conn(size=7, depth=32, stride=2, fmapsize=144)
            
            string kernel = cfg[0];

            int size = 0, fmapsize = 0, depth = 0, stride = 1, bias = 1; 
            string activation = "relu";

            for (int i = 1; i < cfg.Length; i++)
            {
                tkn = cfg[i].Split(new char[] { '=' }, StringSplitOptions.RemoveEmptyEntries);
                switch (tkn[0])
                {
                    case "depth": // number of filters
                        depth = int.Parse(tkn[1]);
                        break;
                    case "fmapsize": // size of target fmaps
                        fmapsize = int.Parse(tkn[1]);
                        break;
                    case "stride": // stride
                        stride = int.Parse(tkn[1]);
                        break;
                    case "size": // size
                        size = int.Parse(tkn[1]);
                        break;
                    case "bias": // bias
                        bias = int.Parse(tkn[1]);
                        break;
                    case "activation": // activation function
                        activation = tkn[1];
                        break;
                    default:
                        throw new Exception();
                }

            }

            Filter k;
            fMap w = null;
            fMap o = null;
            int number_of_fmaps = input.fMaps.Count;
            int index = 0;
            Console.WriteLine(cfg[0]);
            if (cfg[0] == "conn")
            {
                if ((input.filters[0].GetType()==typeof(Filters.Maxpool)) || (input.filters[0].GetType() == typeof(Filters.Convolution)))
                {
                    //if previous layer is pool layer
                    int size_of_fmaps = input.fMaps[0].value.Length;
                    int number_of_filters = size_of_fmaps * size_of_fmaps * number_of_fmaps;
                    for (int i = 0; i < depth; i++)
                    {

                        o = new fMap(fmapsize, fmapsize);
                        o.Bias = bias;
                        fmaps.Add(o);
                        IList<Filter> fmap_filters = new List<Filter>();

                        for (int j = 0; j < number_of_filters; j++)
                        {
                            k = Filter.Build(cfg[0], j, size, stride);
                            w = new fMap(fmapsize, fmapsize);
                            k.Source = w;
                            k.Target = o;
                            k.Activation = activation;
                            filter.Add(k);
                            fmap_filters.Add(k);
                        }
                        connection.Add(fmap_filters);
                    }
                    //if (input.filters[0].GetType() == typeof(Filters.Maxpool))
                    //{

                    //}
                }
                else
                {
                    int number_of_filters =  number_of_fmaps;
                    for (int i = 0; i < depth; i++)
                    {

                        o = new fMap(fmapsize, fmapsize);
                        o.Bias = bias;
                        fmaps.Add(o);
                        IList<Filter> fmap_filters = new List<Filter>();
                        for (int j = 0; j < number_of_filters; j++)
                        {
                            k = Filter.Build(cfg[0], j, size, stride);
                            //w = new fMap(fmapsize, fmapsize);
                            k.Source = input.fMaps[j];
                            k.Target = o;
                            k.Activation = activation;
                            filter.Add(k);
                            fmap_filters.Add(k);
                        }
                        connection.Add(fmap_filters);
                    }
                }

            }
            else
            {
                    if (number_of_fmaps == 1)
                    {
                        //1 channel image
                        for (int i = 0; i < depth; i++)
                        {
                            k = Filter.Build(cfg[0], i, size, stride);
                            //Source  feature map is input image
                            k.Source = input.fMaps[0];
                            w = new fMap(fmapsize, fmapsize);
                            k.Target = w;
                            fmaps.Add(w);
                            k.bias = bias;
                            k.Activation = activation;
                            filter.Add(k);

                        }
                    }

                    if (number_of_fmaps == 3)
                    {
                    //3 channel image (RGB)
                    for (int i = 0; i < depth; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            k = Filter.Build(cfg[0], j, size, stride);
                            //Source  feature map is input image
                            k.Source = input.fMaps[j];
                            w = new fMap(fmapsize, fmapsize);
                            k.Target = w;
                            fmaps.Add(w);
                            k.bias = bias;
                            k.Activation = activation;
                            filter.Add(k);
                        }
                    }
                    }

                    else
                    {
                    //source feature map is from previous conv or pool layer
                    for (int i = 0; i < depth; i++)
                    {
                        if (index < number_of_fmaps)
                        {
                            k = Filter.Build(cfg[0], i, size, stride);
                            k.Source = input.fMaps[index];
                            w = new fMap(fmapsize, fmapsize);
                            k.Target = w;
                            k.bias = bias;
                            k.Activation = activation;
                            filter.Add(k);
                            fmaps.Add(w);
                            index++;
                        }
                    }
                    }
            }

            return this;
        }
        /// <summary>
        /// returns list of feature maps
        /// </summary>
        public IList<fMap> fMaps
        {
            get { return fmaps; }
        }

        public IList<Filter> filters
        {
            get { return filter; }
        }

        public IList<IList<Filter>> Connections
        {
            get { return connection; }
        }

    }
}