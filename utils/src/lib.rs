pub mod file {
    use std::ffi::{/* OsString,  */ OsStr};
    use std::fs;
    use std::fs::read_to_string;
    use std::path::{Path, PathBuf};

    pub fn load_file_split_lines(
        path: impl AsRef<Path>,
        separator: &str,
    ) -> Option<Vec<Vec<String>>> {
        Some(
            read_to_string(path)
                .ok()?
                .lines()
                .map(|line| line.split(separator).map(String::from).collect())
                .collect(),
        )
    }

    pub fn load_file_split_lines_map<T>(
        path: impl AsRef<Path>,
        separator: &str,
        map: impl Fn(&str) -> T,
    ) -> Option<Vec<Vec<T>>> {
        Some(
            read_to_string(path)
                .ok()?
                .lines()
                .map(|line| line.split(separator).map(&map).collect())
                .collect(),
        )
    }

    pub fn split_lines_map<T>(
        data: &str,
        separator: &str,
        map: impl Fn(&str) -> T,
    ) -> Option<Vec<Vec<T>>> {
        Some(
            data.lines()
                .map(|line| line.split(separator).map(&map).collect())
                .collect(),
        )
    }

    pub fn get_col<T: Clone>(data: &Vec<Vec<T>>, index: usize) -> Vec<T> {
        data.iter().map(|row| row[index].clone()).collect()
    }

    pub fn get_cols<T: Clone, const N: usize>(data: &Vec<Vec<T>>, cols: [usize; N]) -> [Vec<T>; N] {
        core::array::from_fn(|i| get_col(data, cols[i]))
    }

    pub fn load_tabbed_file(path: impl AsRef<Path>) -> Option<Vec<Vec<String>>> {
        load_file_split_lines(path, "\t")
    }

    pub fn load_csv_file(path: impl AsRef<Path>) -> Option<Vec<Vec<String>>> {
        load_file_split_lines(path, ",")
    }

    pub fn list_all_files_in_folder_with_extension(
        path: impl AsRef<Path>,
        extension: impl AsRef<OsStr>,
    ) -> Vec<PathBuf> {
        fs::read_dir(path)
            .unwrap()
            .filter_map(|entry| {
                entry
                    .ok()
                    .filter(|p| p.path().extension() == Some(extension.as_ref()))
                    .map(|e| e.path().to_path_buf())
            })
            .collect()
    }

    /* Returns an array of tuples (filename, data) */
    // pub fn load_all_files_in_folder(path: impl AsRef<Path>, extension: impl AsRef<OsStr>) -> Vec<(OsString, Vec<Vec<String>>)> {
    //     list_all_files_in_folder_with_extension(path, extension)
    //         .iter()
    //         .filter_map(|path| load_file_split_lines(path, "\t").map(|data| (path.file_name().expect("File name should be there").to_owned(), data)))
    //         .collect()
    // }
}
