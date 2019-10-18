using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    public interface IReader
    {
        fMap Source { set; }

        int Size { get; }

        int Stride { get; }
    }
}