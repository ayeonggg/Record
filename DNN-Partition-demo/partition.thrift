service Partition { 
    string partition(1: map<string, binary> file, 2: i32 ep, 3: i32 pp, 4: string cORs)
}