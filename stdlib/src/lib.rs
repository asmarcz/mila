use std::ffi::{c_char, CStr};

#[no_mangle]
pub extern "C" fn mila_write(int: i64) {
    print!("{}", int);
}

#[no_mangle]
pub extern "C" fn mila_writeln(int: i64) {
    println!("{}", int);
}

#[no_mangle]
pub extern "C" fn mila_dbl_write(double: f64) {
    print!("{}", double);
}

#[no_mangle]
pub extern "C" fn mila_dbl_writeln(double: f64) {
    println!("{}", double);
}

#[no_mangle]
pub extern "C" fn mila_str_write(ptr: *const c_char) {
    unsafe {
        print!("{}", CStr::from_ptr(ptr).to_str().unwrap());
    }
}

#[no_mangle]
pub extern "C" fn mila_str_writeln(ptr: *const c_char) {
    unsafe {
        println!("{}", CStr::from_ptr(ptr).to_str().unwrap());
    }
}
