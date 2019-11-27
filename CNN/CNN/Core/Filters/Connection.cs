using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core.Filters
{
    public class Connection : Filter
    {
        public Connection(int? id, int size, int stride, double stdev, double scale)
            : base(id, size, stride, stdev, scale) { }
    }
}
