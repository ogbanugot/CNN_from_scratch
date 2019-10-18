using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core.Filters
{
    public class Convolution : Filter
    {
        public Convolution(int? id, int size, int stride)
            : base(id, size, stride) { }

        public int[] forward(int image, int filter, int bias, int stride)
        {
            //performs forward convolution
            return null;
            //returns output feature maps
        }

        public int[] backward(int conv_prev, int conv_in, int filt, int stride)
        {
            //backprop through convolutional layer
            return null;
            //returns dout, dbias, dfilt
        }
    }
}
